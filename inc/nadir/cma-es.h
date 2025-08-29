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
          : Minimizer(fnc, pars), _mp(mp), _mp0(mp), _p0(pars) {};

      void ChangeMetaParameters(MetaParameters mp)
      {
         _mp = mp;
      }

      STATUS minimize() override;

      STATUS ipop_minimize(IPOP_MetaParameters ipop_mp);

   private:
      /// Only free metaparameter
      MetaParameters _mp, _mp0;
      Ran2 ran2;

      void _reset() override
      {
         ran2.seed(-2);
         _mp = _mp0;
         _p0 = _parameters;
      }

      inline void _fill_random_vec(Eigen::VectorXd &v)
      {
         for (long int i = 0; i < v.size(); i++) {
            v(i) = ran2.normal();
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