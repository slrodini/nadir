#ifndef NADIR_MINIMIZER_H
#define NADIR_MINIMIZER_H

#include <vector>
#include <Eigen/Core>

namespace nadir
{
class Minimizer
{
   public:
      Minimizer(std::vector<double *> pars = {}) : _parameters(std::move(pars))
      {
         _gradient.resize(_parameters.size());
         t = 0;
      }

      virtual void step() = 0;

      virtual void AddParameter(double *p)
      {
         _parameters.emplace_back(p);
         _gradient.resize(_parameters.size());
      }

   protected:
      std::vector<double *> _parameters;
      Eigen::VectorXd _gradient;
      size_t t;
};
} // namespace nadir

#endif // NADIR_MINIMIZER_H