#include "nadir/bfgs.h"

namespace nadir
{

void BFGS::_reset()
{
   auto n     = _parameters.size();
   _gt        = Eigen::VectorXd::Zero(n);
   _gt_new    = Eigen::VectorXd::Zero(n);
   _par_trial = Eigen::VectorXd::Zero(n);
   _p         = Eigen::VectorXd::Zero(n);
   H          = Eigen::MatrixXd::Identity(n, n);
   Hinv       = Eigen::MatrixXd::Identity(n, n);
}

BFGS::BFGS(MetaParameters mp, NadirCostFunction &fnc, Eigen::VectorXd pars)
    : Minimizer(fnc, pars), _mp(mp)
{
   _reset();
}

Minimizer::STATUS BFGS::minimize()
{

   /// Temporary variables
   Eigen::VectorXd _s  = Eigen::VectorXd::Zero(_parameters.size());
   Eigen::VectorXd _y  = Eigen::VectorXd::Zero(_parameters.size());
   Eigen::VectorXd _Hs = Eigen::VectorXd::Zero(_parameters.size());

   Eigen::MatrixXd _Id = Eigen::MatrixXd::Identity(_parameters.size(), _parameters.size());
   Eigen::MatrixXd _TM = Eigen::MatrixXd::Identity(_parameters.size(), _parameters.size());

   Minimizer::STATUS status = Minimizer::STATUS::SUCCESS;

   double f_old = 0, f_new = 0.;
   _cost(_parameters, f_old, _gt);

   for (size_t t = 1; t < _mp.max_it; t++) {

      _p = -Hinv * _gt;

      double alpha = _wolfe_ls(f_old);

      _s = alpha * _p;
      _parameters += _s;
      _cost(_parameters, f_new, _gt_new);

      _y = _gt_new - _gt;

      double y_s    = _y.dot(_s);
      _Hs           = H * _s;
      double sT_H_s = _s.dot(_Hs);

      /// Powell damping
      if (y_s <= 1.0e-14) {
         switch (_mp.dam_t) {

         case DAMPING_TYPE::POWELL: {
            double theta = 0;
            if (y_s < 0.2 * sT_H_s) {
               theta = 0.8 * sT_H_s / (sT_H_s - y_s);
            }

            _y = theta * _y + (1. - theta) * _Hs;
            // Recompute y.s with damped y
            y_s = _y.dot(_s);
            if (y_s < 1.0e-10) {
               std::cerr << "WARNING: y.s too small in BFGS after Powell damping. ";
               std::cerr << "Trying to recover." << std::endl;
               _y += 1.0e-5 * _s;
               y_s = _y.dot(_s);
            }
            break;
         }
         case DAMPING_TYPE::M_BFGS: {
            double delta = 1.0e-8;
            _y += _s * (delta - y_s / _s.squaredNorm());
            y_s = _y.dot(_s);
            break;
         }
         case DAMPING_TYPE::NONE:
         default:
            return Minimizer::STATUS::FAILURE;
         }
      }

      double rho = 1.0 / y_s;

      H    = H - (_Hs * _Hs.transpose()) / sT_H_s + rho * _y * _y.transpose();
      _TM  = (_Id - rho * _s * _y.transpose());
      Hinv = rho * _s * _s.transpose() + _TM * Hinv * _TM.transpose();

      const double lambda = 1e-4;

      H    = (1 - lambda) * H + lambda * _Id;
      Hinv = (1 - lambda) * Hinv + lambda * _Id;
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

double BFGS::_wolfe_ls(double f_val)
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

double BFGS::_cu_interpolate(double a0, double f0, double g0, double a1, double f1, double g1)
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

double BFGS::_zoom(double f_0, double g_0, double a_lo, double f_lo, double g_lo, double a_hi,
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

} // namespace nadir