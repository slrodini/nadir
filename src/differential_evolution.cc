#include "nadir/differential_evolution.h"
#include "nadir/ran2.h"
#include <functional>
#include <numeric>

namespace nadir
{

void DiffEvolution::_check_meta_parameters()
{

   if (_mp.NP < 4)
      throw std::invalid_argument(
          "DiffEvolution meta-parameter NP (population size) must be >= 4");
   if (_mp.CR <= 0 || _mp.CR >= 1)
      throw std::invalid_argument(
          "DiffEvolution meta-parameter CR (crossover probability) must be in (0, 1)");

   if (_mp.F <= 0 || _mp.F >= 2)
      throw std::invalid_argument(
          "DiffEvolution meta-parameter F (differential weight) must be in (0, 2)");

   if (std::fabs(_mp.width) < 1.0e-2)
      throw std::invalid_argument("DiffEvolution meta-parameter width (Gaussian std for initial "
                                  "randomization) must be in > 0.01 (in absolute value)");

   if (_mp.max_iter < 100)
      throw std::invalid_argument("DiffEvolution meta-parameter max_iter must be >=100");
}

Minimizer::STATUS DiffEvolution::minimize()
{

   std::vector<Eigen::VectorXd> _population_old; //!< Population of parameters
   std::vector<Eigen::VectorXd> _population_new; //!< Population of parameters
   std::vector<double> _costs_old; //!< Values of cost functions per parameter set in population
   std::vector<double> _costs_new; //!< Values of cost functions per parameter set in population

   for (size_t i = 0; i < _mp.NP; i++) {
      _population_old.push_back(Eigen::VectorXd::Zero(_parameters.size()));
      for (long int k = 0; k < _parameters.size(); k++) {
         _population_old[i](k) = _parameters(k) + _mp.width * _random_normal();
      }
      double tmp = 0;
      _fnc.get().Evaluate(_population_old[i], tmp);
      _costs_old.push_back(tmp);
      _costs_new.push_back(tmp);
   }
   _population_new = _population_old;

   STATUS status = STATUS::SUCCESS;

   using vec_p = std::vector<Eigen::VectorXd> *;
   using vec_d = std::vector<double> *;

   vec_p _old = &_population_old;
   vec_p _new = &_population_new;

   vec_d _c_old = &_costs_old;
   vec_d _c_new = &_costs_new;

   size_t n_par = static_cast<size_t>(_parameters.size());

   for (size_t t = 0; t < _mp.max_iter; t++) {

      for (size_t k = 0; k < (*_old).size(); k++) {
         size_t ia, ib, ic;
         _get_a_b_c_index(k, ia, ib, ic);
         size_t R = _random_uint(n_par);

         for (size_t j = 0; j < n_par; j++) {
            if (j == R || _random_uniform() < _mp.CR) {
               (*_new)[k](j) = (*_old)[ia](j) + _mp.F * ((*_old)[ib](j) - (*_old)[ic](j));
            } else {
               (*_new)[k](j) = (*_old)[k](j);
            }
         }
         double f_new;
         _fnc.get().Evaluate((*_new)[k], f_new);
         if (f_new <= (*_c_old)[k]) {
            (*_c_new)[k] = f_new;
         } else {
            // Remove the changes
            (*_new)[k]   = (*_old)[k];
            (*_c_new)[k] = (*_c_old)[k];
         }
      }

      // Swap the old and new populations, effectively updating the state
      vec_p tmp = _old;
      _old      = _new;
      _new      = tmp;

      vec_d c_tmp = _c_old;
      _c_old      = _c_new;
      _c_new      = c_tmp;

      // A bit wasteful searching the best set of parameters everytime, but so be it
      size_t best_ic = _get_best_parameters(_old, _c_old);
      _buffer << "- {Iteration: " << t << ", Best cost: " << (*_c_old)[best_ic];
      _buffer << "}" << std::endl;

      if (_mp.real_time_progress) {
         std::cout << "- {Iteration: " << t << ", Best cost: " << (*_c_old)[best_ic];
         std::cout << "}" << std::endl;
      }

      if (_fnc_callback) {
         if (!_fnc_callback->get().Evaluate(_parameters)) {
            return STATUS::ABORT;
         }
      }
   }

   return status;
}

void DiffEvolution::_get_a_b_c_index(size_t x, size_t &a, size_t &b, size_t &c)
{
   size_t n = _mp.NP;
   a        = x;
   while (a == x) {
      a = _random_uint(n);
   }

   b = x;
   while (b == x || b == a) {
      b = _random_uint(n);
   }

   c = x;
   while (c == x || c == a || c == b) {
      c = _random_uint(n);
   }
}

size_t DiffEvolution::_get_best_parameters(std::vector<Eigen::VectorXd> *pop,
                                           std::vector<double> *costs)
{
   double m = (*costs)[0];
   size_t k = 0;
   for (size_t i = 1; i < (*costs).size(); i++) {
      if ((*costs)[i] < m) {
         m = (*costs)[i];
         k = i;
      }
   }
   _parameters = (*pop)[k];
   return k;
}

// =================================================================================================

jSOa::STATUS jSOa::minimize()
{

   std::vector<Eigen::VectorXd> _population_old; //!< Population of parameters
   std::vector<Eigen::VectorXd> _population_new; //!< Population of parameters
   std::vector<double> _costs_old; //!< Values of cost functions per parameter set in population
   std::vector<double> _costs_new; //!< Values of cost functions per parameter set in population

   for (size_t i = 0; i < _mp.NP_ini; i++) {
      _population_old.push_back(Eigen::VectorXd::Zero(_parameters.size()));
      for (long int k = 0; k < _parameters.size(); k++) {
         _population_old[i](k) = _parameters(k) + _mp.width * _random_normal();
      }
      double tmp = 0;
      _fnc.get().Evaluate(_population_old[i], tmp);
      _costs_old.push_back(tmp);
      _costs_new.push_back(tmp);
   }
   _population_new = _population_old;

   STATUS status = STATUS::SUCCESS;

   using vec_p = std::vector<Eigen::VectorXd> *;
   using vec_d = std::vector<double> *;

   vec_p _old = &_population_old;
   vec_p _new = &_population_new;

   vec_d _c_old = &_costs_old;
   vec_d _c_new = &_costs_new;

   size_t n_par = static_cast<size_t>(_parameters.size());

   std::vector<double> S_CR(_mp.NP_ini, 0);
   std::vector<double> S_F(_mp.NP_ini, 0);

   std::vector<double> M_CR(_mp.H, 0.5);
   std::vector<double> M_F(_mp.H, 0.8);

   size_t k = 0;

   size_t NP = _mp.NP_ini;
   std::vector<Eigen::VectorXd> archive;
   std::vector<double> archive_cost;
   std::vector<size_t> archive_indexes;
   size_t A_size = NP;

   auto get_r1_r2 = [&NP, &archive](size_t x) -> std::pair<size_t, size_t> {
      size_t a, b;
      a = x;
      while (a == x) {
         a = _random_uint(NP);
      }

      b = x;
      while (b == x || b == a) {
         b = _random_uint(NP + archive.size());
      }
      return {a, b};
   };

   std::vector<size_t> costs_indexes(NP);
   std::vector<double> _wk(NP, 0.);
   std::vector<double> _improv(NP, 0.);

   size_t fnc_eval = 0;

   auto push_progressive = [&](const Eigen::VectorXd &parent, double fparent) {
      if (archive.size() < A_size) {
         archive.push_back(parent);
         archive_cost.push_back(fparent);
         return;
      }

      size_t A    = archive.size();
      size_t tail = std::max<size_t>(1, (size_t)std::ceil(_mp.ap * A));

      archive_indexes.resize(A);
      std::iota(archive_indexes.begin(), archive_indexes.end(), 0);
      std::stable_sort(archive_indexes.begin(), archive_indexes.end(), [&](size_t u, size_t v) {
         return archive_cost[u] < archive_cost[v];
      });

      size_t repl_idx        = archive_indexes[A - tail + _random_uint(tail)];
      archive[repl_idx]      = parent;
      archive_cost[repl_idx] = fparent;
   };

   size_t max_fnc_eval = _mp.NP_ini * _mp.max_iter;

   for (size_t t = 0; t < _mp.max_iter; t++) {
      // double p      = _mp.p_min + t * (double)(_mp.p_max - _mp.p_min) / ((double)_mp.max_iter);
      double p = _mp.p_min + fnc_eval * (double)(_mp.p_max - _mp.p_min) / ((double)max_fnc_eval);

      p             = std::clamp(p, _mp.p_min, _mp.p_max);
      size_t pcount = std::max<size_t>(2, (size_t)std::floor(p * NP));

      S_CR.resize(0);
      S_F.resize(0);
      _improv.resize(0);

      costs_indexes.resize(NP);
      for (size_t i = 0; i < NP; i++) {
         costs_indexes[i] = i;
      }

      std::stable_sort(costs_indexes.begin(), costs_indexes.end(),
                       [&](std::size_t i, std::size_t j) {
                          if (std::isnan((*_c_old)[i])) return false;
                          if (std::isnan((*_c_old)[j])) return true;
                          return (*_c_old)[i] < (*_c_old)[j];
                       });

      for (size_t i = 0; i < NP; i++) {
         size_t ri = _random_uint(_mp.H);
         // if (ri == _mp.H - 1) {
         //    M_CR[ri] = 0.9;
         //    M_F[ri]  = 0.9;
         // }

         double Fi, CRi;

         if (M_CR[ri] < 0) {
            CRi = 0;
         } else {
            CRi = std::clamp(0.1 * _random_normal() + M_CR[ri], 0., 1.);
         }

         if (t < 0.25 * _mp.max_iter) {
            CRi = std::max(CRi, 0.7);
         } else if (t < 0.5 * _mp.max_iter) {
            CRi = std::max(CRi, 0.6);
         }

         do {
            Fi = _random_cauchy(M_F[ri], 0.1);
         } while (Fi <= 0);
         Fi = std::min(Fi, 1.0);
         if (t < 0.6 * _mp.max_iter && Fi > 0.7) {
            Fi = 0.7;
         }

         size_t pbest = costs_indexes[_random_uint(pcount)];

         auto [r1, r2] = get_r1_r2(i);
         double Fw     = Fi;
         if (t < 0.2 * _mp.max_iter) Fw *= 0.7;
         else if (t < 0.4 * _mp.max_iter) Fw *= 0.8;
         else Fw *= 1.2;

         size_t R = _random_uint(n_par);
         if (r2 >= NP) {
            for (size_t j = 0; j < n_par; j++) {
               if (j == R || _random_uniform() < CRi) {
                  (*_new)[i](j) = (*_old)[i](j) + Fw * ((*_old)[pbest](j) - (*_old)[i](j)) +
                                  Fi * ((*_old)[r1](j) - archive[r2 - NP](j));
               } else {
                  (*_new)[i](j) = (*_old)[i](j);
               }
            }
         } else {
            for (size_t j = 0; j < n_par; j++) {
               if (j == R || _random_uniform() < CRi) {
                  (*_new)[i](j) = (*_old)[i](j) + Fw * ((*_old)[pbest](j) - (*_old)[i](j)) +
                                  Fi * ((*_old)[r1](j) - (*_old)[r2](j));
               } else {
                  (*_new)[i](j) = (*_old)[i](j);
               }
            }
         }

         _fnc.get().Evaluate((*_new)[i], (*_c_new)[i]);
         fnc_eval++;

         if ((*_c_new)[i] <= (*_c_old)[i]) {
            if (_mp.impr_archive_eviction) {
               push_progressive((*_old)[i], (*_c_old)[i]);
            } else {
               archive.push_back((*_old)[i]);
               archive_cost.push_back((*_c_old)[i]);
            }
            S_CR.push_back(CRi);
            S_F.push_back(Fi);
            _improv.push_back((*_c_old)[i] - (*_c_new)[i]);
         } else {
            (*_new)[i]   = (*_old)[i];
            (*_c_new)[i] = (*_c_old)[i];
         }
      }
      if (fnc_eval >= max_fnc_eval) {
         _get_best_parameters(_old, _c_old);
         break;
      }

      if (!_improv.empty()) {
         double wsum = std::accumulate(_improv.begin(), _improv.end(), 0.0);
         double numF = 0.0, denF = 0.0, meanCR = 0.0;
         for (size_t s = 0; s < _improv.size(); ++s) {
            double w = _improv[s] / wsum;
            numF += w * S_F[s] * S_F[s];
            denF += w * S_F[s];
            meanCR += w * S_CR[s];
         }
         if (denF > 0.0) M_F[k] = numF / denF;
         M_CR[k] = (meanCR == 0.0) ? -1.0 : meanCR;
         k       = (k + 1) % _mp.H;
      }

      vec_p tmp = _old;
      _old      = _new;
      _new      = tmp;

      vec_d c_tmp = _c_old;
      _c_old      = _c_new;
      _c_new      = c_tmp;

      size_t NP_new =
          (size_t)std::floor(_mp.NP_min + (_mp.NP_ini - _mp.NP_min) *
                                              (1.0 - (double)fnc_eval / (double)max_fnc_eval));
      NP_new = std::max(_mp.NP_min, std::min(NP_new, NP));
      if (NP_new < NP) {
         std::stable_sort(costs_indexes.begin(), costs_indexes.end(),
                          [&](std::size_t i, std::size_t j) {
                             if (std::isnan((*_c_old)[i])) return false;
                             if (std::isnan((*_c_old)[j])) return true;
                             return (*_c_old)[i] < (*_c_old)[j];
                          });
         for (size_t i = 0; i < NP_new; i++) {
            (*_new)[i]   = (*_old)[costs_indexes[i]];
            (*_c_new)[i] = (*_c_old)[costs_indexes[i]];
         }
         (*_new).resize(NP_new);
         (*_old).resize(NP_new);
         (*_c_new).resize(NP_new);
         (*_c_old).resize(NP_new);

         A_size = NP_new;

         if (_mp.impr_archive_eviction) {
            while (archive.size() > A_size) {
               size_t A    = archive.size();
               size_t tail = std::max<size_t>(1, (size_t)std::ceil(_mp.ap * A));

               archive_indexes.resize(A);
               std::iota(archive_indexes.begin(), archive_indexes.end(), 0);
               std::stable_sort(archive_indexes.begin(), archive_indexes.end(),
                                [&](size_t u, size_t v) {
                                   return archive_cost[u] < archive_cost[v];
                                });

               size_t repl = archive_indexes[A - tail + _random_uint(tail)];
               if (repl != A - 1) {
                  archive[repl]      = archive.back();
                  archive_cost[repl] = archive_cost.back();
               }
               archive.pop_back();
               archive_cost.pop_back();
            }
         } else {
            while (archive.size() > A_size) {
               size_t i   = _random_uint(archive.size());
               archive[i] = archive.back();
               archive.pop_back();
            }
         }

         NP = NP_new;

         tmp  = _old;
         _old = _new;
         _new = tmp;

         c_tmp  = _c_old;
         _c_old = _c_new;
         _c_new = c_tmp;
      }

      size_t best_ic = _get_best_parameters(_old, _c_old);
      _buffer << "- {Iteration: " << t;
      _buffer << ", Best cost: " << (*_c_old)[best_ic];
      _buffer << ", NP: " << NP;
      _buffer << "}" << std::endl;

      if (_mp.real_time_progress) {
         std::cerr << "- {Iteration: " << t;
         std::cerr << ", Best cost: " << (*_c_old)[best_ic];
         std::cerr << ", NP: " << NP;
         std::cerr << "}" << std::endl;
      }

      if (_fnc_callback) {
         if (!_fnc_callback->get().Evaluate(_parameters)) {
            return STATUS::ABORT;
         }
      }
   }

   return status;
}

size_t jSOa::_get_best_parameters(std::vector<Eigen::VectorXd> *pop, std::vector<double> *costs)
{
   double m = (*costs)[0];
   size_t k = 0;
   for (size_t i = 1; i < (*costs).size(); i++) {
      if ((*costs)[i] < m) {
         m = (*costs)[i];
         k = i;
      }
   }
   _parameters = (*pop)[k];
   return k;
}

} // namespace nadir