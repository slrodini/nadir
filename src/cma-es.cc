#include "nadir/cma-es.h"

#include <Eigen/Dense>
#include <deque>
#include <functional>

namespace nadir
{

void CMA_ES::_init()
{
   double _alpha_m_mu, _alpha_m_mu_eff, _alpha_m_posdef;
   double _mu_eff_minus;

   _lambda = std::max(4 + static_cast<size_t>(floor(3 * log(_n))), _mp.lambda);
   _sigma  = _mp.sigma;

   _wi.resize(_lambda, 0.);

   _mu = (size_t)floor(_lambda * 0.5);
   // std::cout << _mu << std::endl;
   double s_m = 0, s_m2 = 0;
   double s_p = 0, s_p2 = 0;

   for (size_t i = 0; i < _lambda; i++) {
      _wi[i] = log(0.5 * (1 + _lambda) / ((double)i + 1));
      // std::cout << i << " " << _wi[i] << std::endl;

      if (_wi[i] < 0) {
         s_m += _wi[i];
         s_m2 += _wi[i] * _wi[i];
      } else {
         s_p += _wi[i];
         s_p2 += _wi[i] * _wi[i];
      }
   }

   _mu_eff       = s_p * s_p / s_p2;
   _mu_eff_minus = s_m * s_m / s_m2;

   _c_1 = 2. / (_mu_eff + pow(_n + 1.3, 2));
   _c_c = (4. + _mu_eff / ((double)_n)) / (_n + 4. + 2. * _mu_eff / ((double)_n));
   _c_mu =
       std::min(1. - _c_1, 2. * (0.25 + _mu_eff + 1. / _mu_eff - 2.) / (pow(_n + 2, 2) + _mu_eff));

   // _c_mu = std::min(1. - _c_1, 2. * (_mu_eff + 1. / _mu_eff - 2.) / (pow(_n + 2, 2) + _mu_eff));

   _alpha_m_mu     = 1. + _c_1 / _c_mu;
   _alpha_m_mu_eff = 1. + 2. * _mu_eff_minus / (2. + _mu_eff);
   _alpha_m_posdef = (1. - _c_1 - _c_mu) / (_n * _c_mu);

   for (size_t i = 0; i < _lambda; i++) {
      if (_wi[i] >= 0) {
         _wi[i] /= s_p;
      } else {
         _wi[i] *= std::min(_alpha_m_mu, std::min(_alpha_m_mu_eff, _alpha_m_posdef)) / (-s_m);
      }
   }

   _c_sigma = (_mu_eff + 2.) / (_n + _mu_eff + 5.);
   _d_sigma = 1 + _c_sigma + 2. * std::max(0., sqrt((_mu_eff - 1.) / (_n + 1.)) - 1.);

   _c_m = 1.;

   // // Check
   // s = 0;
   // for (size_t i = 0; i < _mu; i++) {
   //    s += _wi[i];
   // }
   // std::cout << "Sum: " << s << std::endl;

   _w_sum = 0;
   for (size_t i = 0; i < _lambda; i++) {
      _w_sum += _wi[i];
   }
}

CMA_ES::STATUS CMA_ES::minimize()
{

   /// Covariance matrix in the search space and its decomposition
   Eigen::MatrixXd C = Eigen::MatrixXd::Identity(_n, _n);

   /// Vector of mean at each generation
   Eigen::VectorXd m     = _parameters;
   Eigen::VectorXd z_ave = Eigen::VectorXd::Zero(_n);
   std::vector<Eigen::VectorXd> zk;

   for (size_t i = 0; i < _lambda; i++) {
      zk.emplace_back(Eigen::VectorXd::Zero(_n));
   }

   _cached_single_run.fnc_eval = 0;
   size_t t                    = 0;

   double step_size_s = sqrt(_c_sigma * (2. - _c_sigma) * _mu_eff);
   double step_size_c = sqrt(_c_c * (2. - _c_c) * _mu_eff);

   Eigen::MatrixXd B = C;
   Eigen::MatrixXd D = C;

   Eigen::VectorXd p_sigma = Eigen::VectorXd::Zero(_n);
   Eigen::VectorXd p_c     = Eigen::VectorXd::Zero(_n);

   std::vector<double> wi_dot(_lambda);
   for (size_t i = 0; i < _lambda; i++) {
      wi_dot[i] = _wi[i];
   }

   std::vector<double> costs(_lambda, 0.);
   std::vector<size_t> costs_indexes(_lambda);
   for (size_t i = 0; i < _lambda; i++) {
      costs_indexes[i] = i;
   }

   const double en = sqrt((double)_n) * (1.0 - 1. / (4. * _n) + 1. / (21. * _n * _n));

   std::deque<double> history;
   size_t max_histo = 10 + static_cast<size_t>(30. * _n / ((double)_lambda));

   // double sigma_times_max_D = _sigma;
   _cached_single_run.best_par = _parameters;
   _fnc.get().Evaluate(_cached_single_run.best_par, _cached_single_run.fnc_best);

   while (true) {
      // TODO: Extra Status to differentiate the two
      if (t > _mp.max_iter || _cached_single_run.fnc_eval > _mp.max_fnc_eval)
         return STATUS::MAX_IT;

      // Generation, selection and recombination
      for (size_t k = 0; k < _lambda; k++) {
         _fill_random_vec(zk[k]);
         _fnc.get().Evaluate(m + _sigma * B * D * zk[k], costs[k]);
      }

      std::stable_sort(costs_indexes.begin(), costs_indexes.end(),
                       [&](std::size_t i, std::size_t j) {
                          if (std::isnan(costs[i])) return false;
                          if (std::isnan(costs[j])) return true;
                          return costs[i] < costs[j];
                       });

      if (costs[costs_indexes[0]] < _cached_single_run.fnc_best) {
         _parameters = m + _sigma * B * D * zk[costs_indexes[0]];

         _cached_single_run.fnc_best = costs[costs_indexes[0]];

         _cached_single_run.best_par = _parameters;
      }
      history.push_back(costs[costs_indexes[0]]);
      if (history.size() > max_histo) {
         history.pop_front();
      }
      if (history.size() == max_histo) {
         auto [minIt, maxIt] = std::minmax_element(history.begin(), history.end());

         if (std::fabs((*maxIt) - (*minIt)) < 1.0e-14) {
            return STATUS::LOW_DIFF;
         };
      }
      if (_fnc_callback) {
         if (!_fnc_callback->get().Evaluate(_parameters)) {
            return STATUS::ABORT;
         }
      }

      z_ave = Eigen::VectorXd::Zero(_n);
      for (size_t i = 0; i < _mu; i++) {
         z_ave += _wi[i] * zk[costs_indexes[i]];
      }
      m += _c_m * _sigma * B * D * z_ave;

      // Step size control
      p_sigma = (1. - _c_sigma) * p_sigma + step_size_s * B * z_ave;

      double p_sigma_norm = p_sigma.norm();
      _sigma *= exp((_c_sigma / _d_sigma) * (-1. + p_sigma_norm / en));

      double det =
          p_sigma_norm / sqrt(1. - pow(1. - _c_sigma, 2 * (t + 1))) - (1.4 + 2. / (_n + 1.)) * en;

      const double h_sigma       = (det < 0) ? 1 : 0;
      const double delta_h_sigma = (1 - h_sigma) * _c_c * (2. - _c_c);

      p_c = (1.0 - _c_c) * p_c + h_sigma * step_size_c * B * D * z_ave;

      for (size_t i = _mu; i < _lambda; i++) {
         wi_dot[i] = _wi[i] * _n / zk[costs_indexes[i]].squaredNorm();
      }

      // Update Covariance matrix
      C *= (1. + _c_1 * delta_h_sigma - _c_1 - _c_mu * _w_sum);
      C += _c_1 * (p_c * p_c.transpose());
      for (size_t i = 0; i < _lambda; i++) {
         C += _c_mu * wi_dot[i] *
              ((B * D * zk[costs_indexes[i]]) * (B * D * zk[costs_indexes[i]]).transpose());
      }
      C = 0.5 * (C + C.transpose());

      _buffer << "- {Iteration: " << t;
      _buffer << ", Function value: " << costs[costs_indexes[0]];
      _buffer << ", sigma: " << _sigma;
      _buffer << "}" << std::endl;

      if (_mp.real_time_progress) {
         std::cerr << "- {Iteration: " << t;
         std::cerr << ", Function value: " << costs[costs_indexes[0]];
         std::cerr << ", sigma: " << _sigma;
         std::cerr << "}" << std::endl;
      }

      // Update covariance matrix decomp
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(C);
      if (es.info() != Eigen::Success) throw std::runtime_error("eigendecomp failed");

      B = es.eigenvectors();
      D = es.eigenvalues().cwiseMax(0.0).cwiseSqrt().asDiagonal();

      if ((B * D * D * B.transpose() - C).cwiseAbs().maxCoeff() > 1.0e-14) {
         return STATUS::FAILURE;
      }

      // double new_sigma_times_max_D = _sigma * D.diagonal().maxCoeff();
      // if (std::fabs(1. - sigma_times_max_D / new_sigma_times_max_D) > 1.0e+4) {
      //    return STATUS::FAILURE;
      // }

      // sigma_times_max_D = new_sigma_times_max_D;

      if ((D(_n - 1, _n - 1) * D(_n - 1, _n - 1)) / (D(0, 0) * D(0, 0)) > 1.0e14) {
         return STATUS::SUCCESS;
      };

      t++;
      _cached_single_run.fnc_eval += _lambda;
   }

   return STATUS::SUCCESS;
}

CMA_ES::STATUS CMA_ES::ipop_minimize(CMA_ES::IPOP_MetaParameters ipop_mp)
{
   size_t evals_small = 0, evals_large = 0;
   size_t k_large     = 0;
   size_t evals_total = 0;

   double global_best;
   _fnc.get().Evaluate(_parameters, global_best);
   Eigen::VectorXd p_best = _p0;

   // Canonica value for lambda
   size_t lambda_def = 4 + static_cast<size_t>(floor(3 * log(_n)));

   size_t t = 0;

   while (evals_total < ipop_mp.max_fnc_eval) {

      if (t > ipop_mp.max_iter || evals_total > ipop_mp.max_fnc_eval) break;

      const bool run_large = (evals_large <= evals_small);

      if (run_large) {
         _mp.lambda = std::lround(lambda_def * std::pow(ipop_mp.pop_growth, k_large));
         _mp.sigma  = ipop_mp.sigma_ref;

         for (long int i = 0; i < _p0.size(); i++) {
            _parameters(i) += _mp.sigma * _random_normal();
         }
      } else {
         const size_t lambda_large = (size_t)std::lround(
             lambda_def * std::pow(ipop_mp.pop_growth, std::max<size_t>(k_large, 1)));

         // lambda = draw_lambda_small(lambda_def, lambda_large);
         // sigma0 = draw_sigma_small(ipop_mp.sigma_ref);

         double lt = 0.5 * lambda_def *
                     pow((double)lambda_large / (double)lambda_def, pow(_random_uniform(), 2));
         _mp.lambda = std::clamp((size_t)std::floor(lt), lambda_def, lambda_large / 2);

         // sigma0 = ipop_mp.sigma_ref * ((1. - 1.0e-2) * _random_uniform() + 1.0e-2);
         double u  = _random_uniform();
         _mp.sigma = ipop_mp.sigma_ref * std::pow(10.0, -2.0 * u);

         _parameters = p_best;
      }

      _init();
      (void)minimize(); // single run

      evals_total += _cached_single_run.fnc_eval;
      if (run_large) {
         evals_large += _cached_single_run.fnc_eval;
         k_large++;
      } else {
         evals_small += _cached_single_run.fnc_eval;
      }

      if (_cached_single_run.fnc_best < global_best) {
         global_best = _cached_single_run.fnc_best;
         p_best      = _cached_single_run.best_par;
      }

      t++;
   }

   _parameters = p_best;

   return STATUS::SUCCESS;
}

} // namespace nadir