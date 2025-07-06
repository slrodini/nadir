#include <nadir/adam.h>
#include <iostream>
namespace nadir
{
Adam::Adam(std::function<void(Eigen::VectorXd &)> fnc_gradient, std::vector<double *> pars,
           AdamMetaParameters mp)
    : Minimizer(pars), _fnc_gradient(fnc_gradient)
{
   _mt.resize(_parameters.size());
   _vt.resize(_parameters.size());
   _meta_par = mp;
}

void Adam::AddParameter(double *p)
{
   _parameters.emplace_back(p);
   _gradient.resize(_parameters.size());
   _mt.resize(_parameters.size());
   _vt.resize(_parameters.size());
}

void Adam::step()
{
   t++;
   _fnc_gradient(_gradient);
   _mt = _mt * _meta_par._beta1 + (1 - _meta_par._beta1) * _gradient;
   _vt = _vt * _meta_par._beta2 + (1 - _meta_par._beta2) * _gradient.cwiseProduct(_gradient);

   double alpha_t =
       _meta_par._alpha * sqrt(1. - pow(_meta_par._beta2, t)) / (1. - pow(_meta_par._beta1, t));
   for (size_t i = 0; i < _parameters.size(); i++) {
      *(_parameters[i]) -= alpha_t * (_mt(i) / (sqrt(_vt(i)) + _meta_par._eps));
      std::cout << *(_parameters[i]) << " ";
      std::cout << _gradient(i) << " ";
      std::cout << _mt(i) << " ";
      std::cout << _vt(i) << " ";
      std::cout << alpha_t * (_mt(i) / (sqrt(_vt(i)) + _meta_par._eps)) << "\n";
   }
}
} // namespace nadir