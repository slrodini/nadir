#include "nadir/bfgs.h"

namespace nadir
{

LBFGS::LBFGS(MetaParameters &mp, NadirCostFunction &fnc, Eigen::VectorXd pars)
    : Minimizer(fnc, pars), _mp(mp)
{
   _gt        = Eigen::VectorXd::Zero(pars.size());
   _q         = Eigen::VectorXd::Zero(pars.size());
   _gt_new    = Eigen::VectorXd::Zero(pars.size());
   _par_trial = Eigen::VectorXd::Zero(pars.size());
   _p         = Eigen::VectorXd::Zero(_parameters.size());
}

LBFGS::LBFGS(MetaParameters &mp, NadirCostFunction &fnc, long int n_par)
    : Minimizer(fnc, n_par), _mp(mp)
{
   _gt        = Eigen::VectorXd::Zero(n_par);
   _q         = Eigen::VectorXd::Zero(n_par);
   _gt_new    = Eigen::VectorXd::Zero(n_par);
   _par_trial = Eigen::VectorXd::Zero(_parameters.size());
   _p         = Eigen::VectorXd::Zero(_parameters.size());
}

Minimizer::STATUS LBFGS::minimize()
{

   /// Temporary variables
   Eigen::VectorXd _s  = Eigen::VectorXd::Zero(_parameters.size());
   Eigen::VectorXd _y  = Eigen::VectorXd::Zero(_parameters.size());
   Eigen::VectorXd _Hs = Eigen::VectorXd::Zero(_parameters.size());

   Minimizer::STATUS status = Minimizer::STATUS::SUCCESS;

   double f_old = 0, f_new = 0.;
   _cost(_parameters, f_old, _gt);

   for (size_t t = 1; t < _mp.max_it; t++) {

      /// This fills _p from _gt using two loop recursion formula
      if (_s_list.empty()) {
         _p = -_gt;
      } else {
         _two_loop_recursion();
      }
      double alpha = _wolfe_ls(f_old);

      _s = alpha * _p;

      _parameters += _s;
      _cost(_parameters, f_new, _gt_new);

      _y = _gt_new - _gt;

      double y_s = _y.dot(_s);

      if (y_s > 1e-10) {
         if (_s_list.size() == _mp.memory) {
            _s_list.pop_front();
            _y_list.pop_front();
            _rho_list.pop_front();
         }
         _s_list.push_back(_s);
         _y_list.push_back(_y);
         _rho_list.push_back(1.0 / y_s);
      } else {
         std::cerr << "Skipping curvature update: y.s too small." << std::endl;
      }

      // ----- Check for convergence and prepare for next iteration

      if (_fnc_callback) {
         if (!_fnc_callback->get().Evaluate(_parameters)) {
            return STATUS::ABORT;
         }
      }

      double gradient_norm = sqrt(_gt_new.squaredNorm());
      if (gradient_norm < _mp.grad_toll) {
         status = STATUS::SUCCESS;
         break;
      }

      if (std::fabs(f_new - f_old) < _mp.diff_value_toll) {
         status = STATUS::LOW_DIFF;
         break;
      }

      _buffer << "- {Iteration: " << t;
      _buffer << ", Function value: " << f_new;
      _buffer << ", Gradient norm: " << gradient_norm << "}" << std::endl;

      _gt   = _gt_new;
      f_old = f_new;
   }

   return status;
}

double LBFGS::_wolfe_ls(double f_val)
{
   // Can access the gradient _gt, the parameters _parameters,
   // the serach direction _p

   double alpha_trial = _mp.alpha;
   double alpha_prev  = 0.;
   double f_prev      = f_val;
   double g_prev      = _gt.dot(_p);
   double f_zero      = f_val;
   double g_zero      = g_prev;

   _par_trial = _parameters + alpha_trial * _p;
   double f_trial;
   _cost(_par_trial, f_trial, _gt_new);
   double g_trial = _gt_new.dot(_p);

   size_t max_line_iter = 20;
   for (size_t i = 0; i < max_line_iter; i++) {
      // if (f_trial > f_val + _mp.c1 * alpha_trial * g_prev || (i > 0 && f_trial >= f_prev)) {
      if (f_trial > f_val + _mp.c1 * alpha_trial * g_zero || (i > 0 && f_trial >= f_prev)) {
         return _zoom(f_zero, g_zero, alpha_prev, f_prev, g_prev, alpha_trial, f_trial, g_trial);
      }
      // if (std::fabs(g_trial) <= -_mp.c2 * g_prev) {
      if (std::fabs(g_trial) <= -_mp.c2 * g_zero) {
         return alpha_trial;
      }
      if (g_trial >= 0) {
         return _zoom(f_zero, g_zero, alpha_trial, f_trial, g_trial, alpha_prev, f_prev, g_prev);
      }

      double alpha_next =
          _cu_interpolate(alpha_prev, f_prev, g_prev, alpha_trial, f_trial, g_trial);

      alpha_prev = alpha_trial;
      f_prev     = f_trial;
      g_prev     = g_trial;

      alpha_trial = alpha_next;
      _par_trial  = _parameters + alpha_trial * _p;
      _cost(_par_trial, f_trial, _gt_new);
      g_trial = _gt_new.dot(_p);
   }

   return alpha_trial;
}

double LBFGS::_cu_interpolate(double a0, double f0, double g0, double a1, double f1, double g1)
{
   double d1   = g0 + g1 - 3. * (f0 - f1) / (a0 - a1);
   double disc = d1 * d1 - g0 * g1;
   if (disc < 0.0) {
      return 0.5 * (a0 + a1);
   }
   double d2  = sqrt(disc);
   double den = g1 - g0 + 2. * d2;

   if (std::fabs(den) < 1.0e-14) return 0.5 * (a0 + a1);

   double a_new = a1 - (a1 - a0) * (d2 - d1 + g1) / den;

   double lower = std::min(a0, a1);
   double upper = std::max(a0, a1);
   if (a_new < lower || a_new > upper) {
      return 0.5 * (a0 + a1); // fallback to bisection
   }

   return a_new;
}

double LBFGS::_zoom(double f_0, double g_0, double a_lo, double f_lo, double g_lo, double a_hi,
                    double f_hi, double g_hi)
{

   double a_trial, f_trial, g_trial;

   size_t max_zoom_iter = 10;
   for (size_t i = 0; i < max_zoom_iter; i++) {
      a_trial = _cu_interpolate(a_lo, f_lo, g_lo, a_hi, f_hi, g_hi);

      _par_trial = _parameters + a_trial * _p;
      _cost(_par_trial, f_trial, _gt_new);
      g_trial = _gt_new.dot(_p);

      // if (f_trial > f_lo + _mp.c1 * a_trial * g_lo || f_trial >= f_lo) {
      if (f_trial > f_0 + _mp.c1 * a_trial * g_0 || f_trial >= f_lo) {
         // Overshot
         a_hi = a_trial;
         f_hi = f_trial;
         g_hi = g_trial;
      } else {
         // if (std::fabs(g_trial) <= -_mp.c2 * g_lo) return a_trial;
         if (std::fabs(g_trial) <= -_mp.c2 * g_0) return a_trial;

         if (g_trial * (a_hi - a_lo) >= 0) {
            a_hi = a_lo;
            f_hi = f_lo;
            g_hi = g_lo;
         }

         a_lo = a_trial;
         f_lo = f_trial;
         g_lo = g_trial;
      }
   }
   return a_trial;
}

void LBFGS::_two_loop_recursion()
{
   size_t k = _s_list.size();
   std::vector<double> alpha(k);
   _q = _gt;

   for (int i = k - 1; i >= 0; --i) {
      alpha[i] = _rho_list[i] * _s_list[i].dot(_q);
      _q -= alpha[i] * _y_list[i];
   }

   // double scaling = (_y_list.back().dot(_y_list.back()) / _y_list.back().dot(_s_list.back()));
   double scaling = (_y_list.back().dot(_s_list.back()) / _y_list.back().dot(_y_list.back()));

   _p = scaling * _q;

   for (size_t i = 0; i < k; ++i) {
      double beta = _rho_list[i] * _y_list[i].dot(_p);
      _p += _s_list[i] * (alpha[i] - beta);
   }

   _p *= -1;
}

} // namespace nadir