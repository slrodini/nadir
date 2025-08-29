#include "nadir/lamb.h"

namespace nadir
{
// =================================================================================================
Lamb::Lamb(const MetaParameters &mp, NadirCostFunction &fnc, Eigen::VectorXd pars)
    : Minimizer(fnc, pars)
{

   _scheduler = [](size_t) {
      return 1.;
   };
   _mp = mp;

   // TODO: add sanity checks on mp.par_blocks
   _mp.par_blocks.push_back(pars.size());
}

// =================================================================================================
Lamb::STATUS Lamb::minimize()
{
   Eigen::VectorXd _gt = Eigen::VectorXd::Zero(_parameters.size());
   Eigen::VectorXd _mt = Eigen::VectorXd::Zero(_parameters.size());
   Eigen::VectorXd _vt = Eigen::VectorXd::Zero(_parameters.size());
   Eigen::VectorXd _rt = Eigen::VectorXd::Zero(_parameters.size());

   double f_new = 0., f_old = 0.;

   STATUS status = STATUS::SUCCESS;
   size_t t      = 0;
   _fnc.get().Evaluate(_parameters, f_old, _gt);
   while (t < _mp.max_it) {

      t++;
      _mt = _mt * _mp.beta1 + (1 - _mp.beta1) * _gt;
      _vt = _vt * _mp.beta2 + (1 - _mp.beta2) * _gt.cwiseProduct(_gt);

      double bias_corr_vt = sqrt(1. - pow_n(_mp.beta2, t));

      double alpha_t = _scheduler(t) * _mp.alpha * bias_corr_vt / (1. - pow_n(_mp.beta1, t));

      _rt = (_mt.array() / (_vt.array().sqrt() + _mp.eps));
      for (size_t i = 0; i < _mp.par_blocks.size() - 1; i++) {
         size_t len = _mp.par_blocks[i + 1] - _mp.par_blocks[i];

         Eigen::VectorXd ri = _rt.segment(_mp.par_blocks[i], len);
         Eigen::VectorXd xi = _parameters.segment(_mp.par_blocks[i], len);

         double scaling = -alpha_t * _mp.phi(xi.norm()) / ((ri + _mp.lambda * xi).norm());
         for (long int j = 0; j < static_cast<long int>(len); j++) {
            _parameters(j + _mp.par_blocks[i]) += scaling * (_rt(j) + _mp.lambda * xi(j));
         }
      }

      // _parameters = _parameters.array() - alpha_t * (_mt.array() / (_vt.array().sqrt() +
      // _mp.eps));

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
      if (_mp.real_time_progress) {
         std::cerr << "- {Iteration: " << t << ", Function value: " << f_new;
         std::cerr << ", Gradient norm: " << gradient_norm << "}" << std::endl;
      }
   }
   if (t == _mp.max_it) status = STATUS::MAX_IT;
   return status;
}
} // namespace nadir