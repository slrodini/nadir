#include "nadir/levenberg_marquardt.h"

namespace nadir
{

LevMarMinimizer::LevMarMinimizer(MetaParameters &mp, NadirLevMarCostFunction &fnc,
                                 Eigen::VectorXd pars)
    : _mp(mp), _fnc(fnc), _parameters(pars)
{
   npar = _parameters.size();
   nres = _fnc.get().ResidualNumber();

   Jacobian     = Eigen::MatrixXd::Zero(nres + npar, npar);
   residual     = Eigen::VectorXd::Zero(nres);
   residual_new = Eigen::VectorXd::Zero(nres);
   std::cout << nres << " " << npar << std::endl;
   std::cout << Jacobian.rows() << " " << Jacobian.cols() << std::endl;

   if (_mp.eta2 >= _mp.eta1) throw std::invalid_argument("Eta2 must be < Eta1");

   /// TODO: extra sanity checks
}

LevMarMinimizer::STATUS LevMarMinimizer::minimize()
{
   STATUS status = STATUS::SUCCESS;
   size_t t;

   Eigen::VectorXd p   = Eigen::VectorXd::Zero(_parameters.size());
   Eigen::MatrixXd _Id = Eigen::MatrixXd::Identity(npar, npar);

   for (t = 0; t < _mp.max_iterations; t++) {
      auto TB = Jacobian.topRows(nres);    // (nres x npar)
      auto BB = Jacobian.bottomRows(npar); // (npar x npar)

      _fnc.get().Evaluate(_parameters, residual, &TB);
      f_old = residual.array().square().sum();

      // Computes the D(x) matrix as the square root of the diagonal of J J^T
      // BB.diagonal() = (TB.array().square().colwise().sum()).sqrt() / sqrt(_mp.mu);
      BB.diagonal() = (TB.array().square().colwise().sum()).sqrt() * _mp.scaling_function(_mp.mu);

      // Eigen::MatrixXd JtJ(_parameters.size(), _parameters.size());
      // JtJ.selfadjointView<Eigen::Upper>().rankUpdate(Jacobian.transpose());

      // Add small regularization to avoid singular Cholesky
      Eigen::LLT<Eigen::MatrixXd> llt(Jacobian.transpose() * Jacobian + 1.0e-12 * _Id);
      if (llt.info() != Eigen::Success) {
         std::cerr << "Cholesky decomposition failed." << std::endl;
         /// Try to increase the regulator
         Eigen::LLT<Eigen::MatrixXd> llt2(Jacobian.transpose() * Jacobian + 1.0e-6 * _Id);
         if (llt2.info() != Eigen::Success) {
            /// J^T J is too singular
            return STATUS::FAILURE;
         } else {
            /// Ok, we can use this step
            p = llt2.solve(-TB.transpose() * residual);
         }

      } else {
         // Solve (M^T M) p = g
         p = llt.solve(-TB.transpose() * residual);
      }

      // Solve (M^T M) p = g
      Eigen::VectorXd p = llt.solve(-TB.transpose() * residual);

      double tmp = (TB * p + residual).array().square().sum();

      _parameters += p;
      _fnc.get().Evaluate(_parameters, residual_new, nullptr);
      f_new = residual_new.array().square().sum();

      double rho = (f_new - f_old) / (tmp - f_old);

      if (rho < _mp.eps) _parameters -= p;
      if (rho > _mp.eta1) _mp.mu *= 2;
      else if (rho < _mp.eta2) _mp.mu *= 0.5;

      if (_fnc_callback) {
         if (!_fnc_callback->get().Evaluate(_parameters)) {
            return STATUS::ABORT;
         }
      }

      double gradient_norm = sqrt(2.0) * (TB.transpose() * residual_new).norm();
      if (gradient_norm < _mp.grad_toll) {
         status = STATUS::SUCCESS;
         break;
      }

      if (std::fabs(f_new - f_old) < _mp.diff_value_toll) {
         status = STATUS::LOW_DIFF;
         break;
      }

      _buffer << "- {Iteration: " << t << ", Function value: " << f_new;
      _buffer << ", Gradient norm: " << gradient_norm << "}" << std::endl;
      if (_mp.real_time_progress) {
         std::cerr << "- {Iteration: " << t << ", Function value: " << f_new;
         std::cerr << ", Gradient norm: " << gradient_norm << "}" << std::endl;
      }
   }
   if (t == _mp.max_iterations) status = STATUS::MAX_IT;
   return status;
}

} // namespace nadir