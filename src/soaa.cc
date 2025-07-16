#include <nadir/soaa.h>

namespace nadir
{
// =================================================================================================
SOAA::SOAA(const MetaParameters &mp, NadirCostFunction &fnc, Eigen::VectorXd pars)
    : Minimizer(fnc, pars)
{
   _gt        = Eigen::VectorXd::Zero(pars.size());
   _mt        = Eigen::VectorXd::Zero(pars.size());
   _vt        = Eigen::VectorXd::Zero(pars.size());
   _mt_hat    = Eigen::VectorXd::Zero(pars.size());
   _vt_hat    = Eigen::VectorXd::Zero(pars.size());
   _Ft        = Eigen::VectorXd::Zero(pars.size());
   _scheduler = [](size_t) {
      return 1.;
   };
   _mp    = mp;
   _dt    = 1.;
   _l_ave = 0.;
   _pr    = 0.;
}

// =================================================================================================
SOAA::SOAA(const MetaParameters &mp, NadirCostFunction &fnc, long int n_par)
    : Minimizer(fnc, n_par)
{
   _parameters = Eigen::VectorXd::Zero(n_par);
   _gt         = Eigen::VectorXd::Zero(n_par);
   _mt         = Eigen::VectorXd::Zero(n_par);
   _vt         = Eigen::VectorXd::Zero(n_par);
   _mt_hat     = Eigen::VectorXd::Zero(n_par);
   _vt_hat     = Eigen::VectorXd::Zero(n_par);
   _Ft         = Eigen::VectorXd::Zero(n_par);
   _scheduler  = [](size_t) {
      return 1.;
   };
   _mp    = mp;
   _dt    = 1.;
   _l_ave = 0.;
   _pr    = 0.;
}

// =================================================================================================
SOAA::STATUS SOAA::minimize()
{
   STATUS status = STATUS::SUCCESS;
   size_t t      = 0;
   _fnc.get().Evaluate(_parameters, f_old, _gt);

   while (t < _mp.max_it) {

      // STEP

      t++;

      double alpha_t = _scheduler(t) * _mp.alpha;

      _mt = _mt * _mp.beta1 + (1 - _mp.beta1) * _gt;
      _vt = _vt * _mp.beta2 + (1 - _mp.beta2) * _gt.cwiseProduct(_gt);

      // Bias correction
      _mt_hat = _mt / (1. - pow_n(_mp.beta1, t));
      _vt_hat = _vt / (1. - pow_n(_mp.beta2, t));

      double rt = 1. + _mt_hat.squaredNorm() / ((_vt_hat.array() + _mp.eps).sum());
      _Ft       = rt * _vt_hat;

      for (long int i = 0; i < _Ft.size(); i++) {
         _Ft(i) = std::max(_dt * _Ft(i), sqrt(_vt_hat(i)));
      }

      _gt = _dt * _mt_hat.array() / (_Ft.array() + _mp.eps);

      _parameters -= alpha_t * _mp.lambda * _parameters + alpha_t * _gt;

      _l_ave       = _mp.beta1 * _l_ave + (1 - _mp.beta1) * f_old;
      double l_hat = _l_ave / (1. - pow_n(_mp.beta1, t));

      double p = static_cast<double>(t - 1) / static_cast<double>(_mp.max_it);

      _dt = std::min(
          1. + pow(_mp.gamma, p),
          std::max(pow(1. - _mp.gamma, p), _dt * (l_hat - f_old) / std::max(_pr, _mp.eps)));

      _pr = alpha_t * (_mt_hat.dot(_gt) - 0.5 * _vt_hat.dot(_gt.cwiseProduct(_gt)));

      _fnc.get().Evaluate(_parameters, f_new, _gt);

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

} // namespace nadir