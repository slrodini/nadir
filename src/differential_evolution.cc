#include "nadir/differential_evolution.h"
#include "nadir/ran2.h"
#include <functional>

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

void DiffEvolution::_init()
{
   if (_population_old.size() > 0) {
      _population_old = std::vector<Eigen::VectorXd>();
      _population_new = std::vector<Eigen::VectorXd>();
      _costs_old      = std::vector<double>();
      _costs_new      = std::vector<double>();
   }
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
}

Minimizer::STATUS DiffEvolution::minimize()
{
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

} // namespace nadir