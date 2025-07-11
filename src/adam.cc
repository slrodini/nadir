#include <nadir/adam.h>

namespace nadir
{
// =================================================================================================
Adam::Adam(NadirCostFunction &fnc, Eigen::VectorXd pars) : Minimizer(fnc, pars)
{
   _gt        = Eigen::VectorXd::Zero(pars.size());
   _mt        = Eigen::VectorXd::Zero(pars.size());
   _vt        = Eigen::VectorXd::Zero(pars.size());
   _vt_hat    = Eigen::VectorXd::Zero(pars.size());
   _mp        = MetaParameters();
   _scheduler = [](size_t) {
      return 1.;
   };
}

// =================================================================================================
Adam::Adam(NadirCostFunction &fnc, long int n_par) : Minimizer(fnc, n_par)
{
   _parameters = Eigen::VectorXd::Zero(n_par);
   _gt         = Eigen::VectorXd::Zero(n_par);
   _mt         = Eigen::VectorXd::Zero(n_par);
   _vt         = Eigen::VectorXd::Zero(n_par);
   _vt_hat     = Eigen::VectorXd::Zero(n_par);
   _mp         = MetaParameters();
   _scheduler  = [](size_t) {
      return 1.;
   };
}

// =================================================================================================
Adam::STATUS Adam::minimize()
{
   STATUS status = STATUS::SUCCESS;
   size_t t      = 0;
   _fnc.get().Evaluate(_parameters, f_old, _gt);

   while (t < _mp.max_it) {

      switch (_mp.variant) {
      case ADAM_VARIANT::CLASSIC:
         t = step(t);
         break;
      case ADAM_VARIANT::AMSGRAD:
         t = step_ams(t);
         break;
      case ADAM_VARIANT::AMSGRAD_V2:
         t = step_ams_v2(t);
         break;
      case ADAM_VARIANT::NADAM:
         t = step_nadam(t);
         break;
      case ADAM_VARIANT::ADAMW:
         t = step_adamw(t);
         break;
      case ADAM_VARIANT::ADABELIEF:
         t = step_adabelief(t);
         break;
      case ADAM_VARIANT::ADABELIEF_W:
         t = step_adabelief_w(t);
         break;
      case ADAM_VARIANT::LION:
         t = step_lion(t);
         break;
      case ADAM_VARIANT::RADAM:
         t = step_radam(t);
         break;
      default:
         throw std::runtime_error("Unknown minimization strategy.");
      }

      if (_fnc_callback) {
         if (!_fnc_callback->get().Evaluate(_parameters)) {
            return STATUS::ABORT;
         }
      }
      double gradient_norm = sqrt(_gt.squaredNorm());
      if (gradient_norm < _mp.grad_toll) {
         status = STATUS::SUCCESS;
         break;
      }

      if (std::fabs(f_new - f_old) < _mp.diff_value_toll) {
         status = STATUS::LOW_DIFF;
         break;
      }
      f_old = f_new;
      _buffer << "- {Iteration: " << t << ", Function value: " << f_new;
      _buffer << ", Gradient norm: " << gradient_norm << "}" << std::endl;
   }
   if (t == _mp.max_it) status = STATUS::MAX_IT;
   return status;
}

// =================================================================================================
// Original algorithm from https://arxiv.org/pdf/1412.6980
size_t Adam::step(size_t t)
{
   t++;
   _mt = _mt * _mp.beta1 + (1 - _mp.beta1) * _gt;
   _vt = _vt * _mp.beta2 + (1 - _mp.beta2) * _gt.cwiseProduct(_gt);

   double bias_corr_vt = sqrt(1. - pow_n(_mp.beta2, t));

   double alpha_t = _scheduler(t) * _mp.alpha * bias_corr_vt / (1. - pow_n(_mp.beta1, t));
   _parameters    = _parameters.array() - alpha_t * (_mt.array() / (_vt.array().sqrt() + _mp.eps));

   _fnc.get().Evaluate(_parameters, f_new, _gt);

   return t;
}

// =================================================================================================
// AMSGrad modification of Adam, from https://arxiv.org/pdf/1904.09237
// Note: since beta1 is assumed constant, this is equivalent to the implementation of
// https://arxiv.org/pdf/1904.03590
size_t Adam::step_ams(size_t t)
{
   t++;
   _mt = _mt * _mp.beta1 + (1 - _mp.beta1) * _gt;
   _vt = _vt * _mp.beta2 + (1 - _mp.beta2) * _gt.cwiseProduct(_gt);

   double alpha_t = _mp.alpha / (1.0 - pow_n(_mp.beta1, t));

   double bias_corr_vt = 1. - pow_n(_mp.beta2, t);

   for (long int i = 0; i < _vt.size(); i++) {
      _vt_hat(i) = std::max(_vt(i) / bias_corr_vt, _vt_hat(i));
   }

   _parameters = _parameters.array() -
                 _scheduler(t) * alpha_t * (_mt.array() / (_vt_hat.array().sqrt() + _mp.eps));

   _fnc.get().Evaluate(_parameters, f_new, _gt);

   return t;
}

size_t Adam::step_ams_v2(size_t t)
{
   t++;
   _mt = _mt * _mp.beta1 + (1 - _mp.beta1) * _gt;
   _vt = _vt * _mp.beta2 + (1 - _mp.beta2) * _gt.cwiseProduct(_gt);

   double alpha_t = _mp.alpha / (1.0 - pow_n(_mp.beta1, t));

   double bias_corr_vt = 1. - pow_n(_mp.beta2, t);

   for (long int i = 0; i < _vt.size(); i++) {
      _vt_hat(i) = std::max(_vt(i), _vt_hat(i)) / bias_corr_vt;
   }

   _parameters = _parameters.array() -
                 _scheduler(t) * alpha_t * (_mt.array() / (_vt_hat.array().sqrt() + _mp.eps));

   _fnc.get().Evaluate(_parameters, f_new, _gt);

   return t;
}

// =================================================================================================
// Implementation of the Nesterov-accellerated Adam variant of
//  https://openreview.net/pdf/OM0jvwB8jIp57ZJjtNEZ.pdf
size_t Adam::step_nadam(size_t t)
{
   t++;
   _mt = _mt * _mp.beta1 + (1 - _mp.beta1) * _gt;
   _vt = _vt * _mp.beta2 + (1 - _mp.beta2) * _gt.cwiseProduct(_gt);

   // Note: here _vt_hat is used in place of _mt_hat from
   // https://openreview.net/pdf/OM0jvwB8jIp57ZJjtNEZ.pdf
   Eigen::VectorXd &_mt_hat = _vt_hat;

   _mt_hat = (_mp.beta1 / (1. - pow_n(_mp.beta1, t + 1))) * _mt +
             ((1. - _mp.beta1) / (1. - pow_n(_mp.beta1, t))) * _gt;

   // Unclear point: some sources have additional beta2 at denominator, others
   // claim that bias correction is done as in Adam.
   // double bias_corr_vt = sqrt((1. - pow_n(_mp.beta2, t)) / _mp.beta2);
   double bias_corr_vt = (1. - pow_n(_mp.beta2, t));

   double alpha_t = _mp.alpha * bias_corr_vt;

   _parameters = _parameters.array() -
                 _scheduler(t) * alpha_t * (_mt_hat.array() / (_vt.array().sqrt() + _mp.eps));

   _fnc.get().Evaluate(_parameters, f_new, _gt);

   return t;
}

// =================================================================================================
// Implementation of Adam with decoupled weight decay from
// https://arxiv.org/pdf/1711.05101
size_t Adam::step_adamw(size_t t)
{
   t++;

   _mt = _mt * _mp.beta1 + (1 - _mp.beta1) * _gt;
   _vt = _vt * _mp.beta2 + (1 - _mp.beta2) * _gt.cwiseProduct(_gt);

   double bias_corr_vt = sqrt(1. - pow_n(_mp.beta2, t));

   double alpha_t = _scheduler(t) * _mp.alpha * bias_corr_vt / (1. - pow_n(_mp.beta1, t));
   _parameters = (_parameters.array() - alpha_t * (_mt.array() / (_vt.array().sqrt() + _mp.eps))) *
                 (1. - alpha_t * _mp.lambda);

   _fnc.get().Evaluate(_parameters, f_new, _gt);

   return t;
}

// =================================================================================================
// Implementation of the AdaBelief algorithm from
// https://arxiv.org/pdf/2010.07468
size_t Adam::step_adabelief(size_t t)
{
   t++;
   _mt = _mt * _mp.beta1 + (1 - _mp.beta1) * _gt;

   // Here _vt_hat is used as a temporary storage
   Eigen::VectorXd &gt_m_mt = _vt_hat;

   gt_m_mt = (_gt - _mt);
   gt_m_mt = gt_m_mt.cwiseProduct(gt_m_mt);
   _vt     = _vt * _mp.beta2 + (1 - _mp.beta2) * gt_m_mt;
   _vt     = _vt.array() + _mp.eps;

   double bias_corr_vt = sqrt(1. - pow_n(_mp.beta2, t));

   double alpha_t = _scheduler(t) * _mp.alpha * bias_corr_vt / (1. - pow_n(_mp.beta1, t));
   _parameters    = _parameters.array() - alpha_t * (_mt.array() / (_vt.array().sqrt() + _mp.eps));

   _fnc.get().Evaluate(_parameters, f_new, _gt);

   return t;
}

// Implementation of the AdaBelief algorithm with decoupled weight decay
size_t Adam::step_adabelief_w(size_t t)
{
   t++;
   _mt = _mt * _mp.beta1 + (1 - _mp.beta1) * _gt;

   // Here _vt_hat is used as a temporary storage
   Eigen::VectorXd &gt_m_mt = _vt_hat;

   gt_m_mt = (_gt - _mt);
   gt_m_mt = gt_m_mt.cwiseProduct(gt_m_mt);
   _vt     = _vt * _mp.beta2 + (1 - _mp.beta2) * gt_m_mt;
   _vt     = _vt.array() + _mp.eps;

   double bias_corr_vt = sqrt(1. - pow_n(_mp.beta2, t));

   double alpha_t = _scheduler(t) * _mp.alpha * bias_corr_vt / (1. - pow_n(_mp.beta1, t));
   _parameters    = _parameters.array() - alpha_t * (_mt.array() / (_vt.array().sqrt() + _mp.eps));
   _parameters *= (1.0 - _mp.lambda * alpha_t);

   _fnc.get().Evaluate(_parameters, f_new, _gt);

   return t;
}

// =================================================================================================
// Implementaion for the Lion algorithm, not exactly a variation of Adam, but close enough
//  https://arxiv.org/pdf/2302.06675
size_t Adam::step_lion(size_t t)
{
   t++;
   Eigen::VectorXd &_ct = _vt;

   _ct = _mt * _mp.beta1 + (1 - _mp.beta1) * _gt;

   double eta_t = _mp.alpha * _scheduler(t);
   _parameters  = _parameters.array() - eta_t * _ct.unaryExpr(sign).array();
   _parameters *= (1. - _mp.lambda * eta_t);

   _fnc.get().Evaluate(_parameters, f_new, _gt);
   _mt = _mt * _mp.beta2 + (1 - _mp.beta2) * _gt;

   return t;
}

// =================================================================================================
// Rectified Adam algorithm from https://arxiv.org/pdf/1908.03265
size_t Adam::step_radam(size_t t)
{
   t++;
   _mt = _mt * _mp.beta1 + (1 - _mp.beta1) * _gt;
   _vt = _vt * _mp.beta2 + (1 - _mp.beta2) * _gt.cwiseProduct(_gt);

   double rho_inf = 2. / (1. - _mp.beta2) - 1.;
   double rho_t   = rho_inf - 2 * t * pow_n(_mp.beta2, t) / (1. - pow_n(_mp.beta2, t));

   if (rho_t > 4) {
      double rt =
          sqrt(rho_inf * (rho_t - 4.) * (rho_t - 2.) / (rho_t * (rho_inf - 4.) * (rho_inf - 2.)));

      double alpha_t = rt * _scheduler(t) * _mp.alpha * sqrt(1. - pow_n(_mp.beta2, t)) /
                       (1. - pow_n(_mp.beta1, t));
      _parameters = _parameters.array() - alpha_t * (_mt.array() / (_vt.array().sqrt() + _mp.eps));

   } else {
      // If we are here, we need to ensure tiny alpha, otherwise we have stability issues
      double alpha_t = _scheduler(t) * std::min(_mp.alpha, 1.0e-3) / (1. - pow_n(_mp.beta1, t));
      _parameters    = _parameters - alpha_t * _mt;
   }

   _fnc.get().Evaluate(_parameters, f_new, _gt);

   return t;
}

} // namespace nadir