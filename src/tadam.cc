#include <nadir/tadam.h>
#include <Eigen/Dense>

namespace nadir
{
// =================================================================================================
TAdam::TAdam(MetaParameters mp, NadirCostFunction &fnc, Eigen::VectorXd pars)
    : Minimizer(fnc, pars)
{

   _scheduler = [](size_t) {
      return 1.;
   };
   _mp = mp;
}

// =================================================================================================
TAdam::STATUS TAdam::minimize()
{

   /// Values of previous and current cost function
   double f_new, f_old;
   double _delta_t, _rho_t;
   double _lt, _lt_hat;
   _delta_t = _mp.delta_0;

   Eigen::VectorXd _old_par  = _parameters;
   Eigen::VectorXd _gt       = Eigen::VectorXd::Zero(_parameters.size());
   Eigen::VectorXd _mt       = Eigen::VectorXd::Zero(_parameters.size());
   Eigen::VectorXd _vt       = Eigen::VectorXd::Zero(_parameters.size());
   Eigen::VectorXd _mt_hat   = Eigen::VectorXd::Zero(_parameters.size());
   Eigen::VectorXd _mtm1_hat = Eigen::VectorXd::Zero(_parameters.size());
   Eigen::VectorXd _vt_hat   = Eigen::VectorXd::Zero(_parameters.size());
   Eigen::VectorXd dp        = Eigen::VectorXd::Zero(_parameters.size());

   Eigen::MatrixXd _vt_bar    = Eigen::MatrixXd::Zero(_parameters.size(), _parameters.size());
   Eigen::MatrixXd _vt_barhat = Eigen::MatrixXd::Zero(_parameters.size(), _parameters.size());
   Eigen::MatrixXd _Ft        = Eigen::MatrixXd::Zero(_parameters.size(), _parameters.size());
   Eigen::MatrixXd _Id        = Eigen::MatrixXd::Identity(_parameters.size(), _parameters.size());

   STATUS status = STATUS::SUCCESS;
   size_t t      = 0;
   _fnc.get().Evaluate(_parameters, f_old, _gt);

   auto _trust_region_radius = [&_delta_t, &_rho_t, &t](size_t mI, double gamma) {
      double p = static_cast<double>(t - 1) / static_cast<double>(mI);
      if (_rho_t < gamma) _delta_t = std::max(1., _delta_t) * pow(1 - gamma, p);
      else if (_rho_t > (1 - gamma)) _delta_t = std::min(1., _delta_t) * (1. + pow(gamma, p));
   };

   while (t < _mp.max_it) {

      // STEP

      t++;
      f_old = f_new;
      _fnc.get().Evaluate(_parameters, f_new, _gt);

      double gradient_norm = sqrt(_gt.squaredNorm());
      if (gradient_norm < _mp.grad_toll) {
         status = STATUS::SUCCESS;
         break;
      }

      if (std::fabs(f_new - f_old) < _mp.diff_value_toll) {
         status = STATUS::LOW_DIFF;
         break;
      }

      double alpha_t = _scheduler(t) * _mp.alpha;

      _mt = _mt * _mp.beta1 + (1 - _mp.beta1) * _gt;
      _vt = _vt * _mp.beta2 + (1 - _mp.beta2) * _gt.cwiseProduct(_gt);

      // Bias correction
      _mt_hat = _mt / (1. - pow_n(_mp.beta1, t));
      _vt_hat = _vt / (1. - pow_n(_mp.beta2, t));

      // In the notation of the paper:
      // g_n = _gt
      // \bar{g_n} = _mt_hat
      // \bar{g_n^2} = _vt_hat
      // \bar{v_n} -> the notation is unclear, but it has to be a vector
      //              hence I take that it should have a component wise mult not shown

      _vt_bar = _vt_bar * _mp.beta2 +
                (1 - _mp.beta2) * ((_gt - _mtm1_hat) * (_gt - _mt_hat).transpose());
      _vt_barhat = _vt_bar / (1. - pow_n(_mp.beta2, t));

      // Now I can update _mtm1
      _mtm1_hat = _mt_hat;

      _Ft = (_mt_hat * _mt_hat.transpose()) + _vt_barhat;

      if (t == 1) _delta_t = _mp.delta_0;
      else {
         double num = _lt_hat - f_new;

         Eigen::VectorXd dp = _old_par - _parameters;

         double den = f_new + _mtm1_hat.dot(dp) + (0.5 / alpha_t) * dp.dot(_Ft * dp);

         _rho_t = num / std::max(_mp.eps, den);
         _trust_region_radius(_mp.max_it, _mp.gamma);
         _lt     = _mp.beta1 * _lt + (1. - _mp.beta1) * f_new;
         _lt_hat = _lt / (1. - pow_n(_mp.beta1, t));
      }

      _old_par = _parameters;
      for (long int i = 0; i < _parameters.size(); i++) {
         _parameters(i) -=
             alpha_t * _mt_hat(i) *
             std::min(1. / (_Ft(i, i) + _mp.eps), _delta_t / (sqrt(_vt_hat(i)) + _mp.eps));
      }

      if (_fnc_callback) {
         if (!_fnc_callback->get().Evaluate(_parameters)) {
            return STATUS::ABORT;
         }
      }

      _buffer << "- {Iteration: " << t << ", Function value: " << f_new;
      _buffer << ", Gradient norm: " << gradient_norm;
      _buffer << ", Trust region radius: " << _delta_t;
      _buffer << "}" << std::endl;

      if (_mp.real_time_progress) {
         std::cerr << "- {Iteration: " << t << ", Function value: " << f_new;
         std::cerr << ", Gradient norm: " << gradient_norm;
         std::cerr << ", Trust region radius: " << _delta_t;
         std::cerr << "}" << std::endl;
      }
   }
   if (t == _mp.max_it) status = STATUS::MAX_IT;
   return status;
}

} // namespace nadir