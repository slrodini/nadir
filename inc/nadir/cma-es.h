#ifndef NADIR_CMA_ES_H
#define NADIR_CMA_ES_H

#include "nadir/abstract_classes.h"
#include "nadir/ran2.h"

#include <Eigen/Core>

namespace nadir
{
class CMA_ES : public Minimizer
{
   public:
      struct MetaParameters {
            double sigma            = 1.;
            size_t lambda           = 0;
            size_t max_fnc_eval     = 1000000;
            size_t max_iter         = 1000;
            bool real_time_progress = false;
      };

      // For multiple run with population growth
      struct IPOP_MetaParameters {
            double sigma_ref        = 1.;
            double pop_growth       = 2.0;
            size_t max_fnc_eval     = 1000000;
            size_t max_iter         = 10;
            bool real_time_progress = false;
      };

   public:
      CMA_ES(MetaParameters mp, NadirCostFunction &fnc, Eigen::VectorXd pars)
          : Minimizer(fnc, pars), _mp(mp), _n(pars.size()), _p0(pars)
      {
         _init();
      };

      CMA_ES(NadirCostFunction &fnc, long int pars = 1) : Minimizer(fnc, pars)
      {
         throw std::runtime_error("Unsupported constructor");
      }

      virtual void SetInitialParameters(const Eigen::VectorXd &pars) override
      {
         _parameters = pars;
      }

      /// Remove ability to set parameters individually after initialization
      virtual void SetInitialParameter(long int, double) override
      {
         throw std::runtime_error("CMA_ES cannot set parameters after initialization.");
      }

      STATUS minimize() override;

      STATUS ipop_minimize(IPOP_MetaParameters ipop_mp);

   private:
      /// Only free metaparameter
      MetaParameters _mp;

      /// Cache some values, for convenience

      /// Number of parameters
      long int _n;

      /// Populaiton size
      size_t _lambda;

      /// Number of positive weights
      size_t _mu;

      double _c_1, _c_mu, _c_c;

      double _c_sigma, _d_sigma;

      double _c_m;

      double _mu_eff;

      double _w_sum;

      double _sigma;

      /// Vector of weights
      std::vector<double> _wi;

      void _init();

      inline void _fill_random_vec(Eigen::VectorXd &v)
      {
         for (long int i = 0; i < v.size(); i++) {
            v(i) = _random_normal();
         }
      }

      struct SingleRunResult {
            double fnc_best;
            size_t fnc_eval;
            Eigen::VectorXd best_par;
      };

      SingleRunResult _cached_single_run;

      Eigen::VectorXd _p0;
};

} // namespace nadir
#endif // NADIR_CMA_ES_H