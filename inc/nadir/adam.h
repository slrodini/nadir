#ifndef NADIR_ADAM_H
#define NADIR_ADAM_H

#include <nadir/minimizer.h>
namespace nadir
{

class Adam : public Minimizer
{

   public:
      struct AdamMetaParameters {
            double _alpha, _beta1, _beta2, _eps;
      };

      Adam(std::function<void(Eigen::VectorXd &)> fnc_gradient, std::vector<double *> pars = {},
           AdamMetaParameters mp = {0.01, 0.9, 0.999, 1.0e-8});
      void AddParameter(double *p) override;
      void step() override;

   private:
      std::function<void(Eigen::VectorXd &)> _fnc_gradient;
      Eigen::VectorXd _mt, _vt;
      AdamMetaParameters _meta_par;
};
} // namespace nadir

#endif