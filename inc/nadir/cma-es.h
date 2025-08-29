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
      struct BIPOP_MetaParameters {
            double sigma_ref        = 1.;
            double pop_growth       = 2.0;
            size_t max_fnc_eval     = 1000000;
            size_t max_iter         = 10;
            bool real_time_progress = false;
      };

   public:
      CMA_ES(MetaParameters mp, BIPOP_MetaParameters bipop_mp, NadirCostFunction &fnc,
             Eigen::VectorXd pars)
          : Minimizer(fnc, pars), _mp(mp), _mp0(mp), _bipop_mp(bipop_mp), _bipop_mp0(bipop_mp),
            _p0(pars) {};

      void ChangeMetaParameters(MetaParameters mp, BIPOP_MetaParameters bipop_mp)
      {
         _mp       = mp;
         _bipop_mp = bipop_mp;
      }

      STATUS single_minimize();

      STATUS minimize() override;

   private:
      /// Only free metaparameter
      MetaParameters _mp, _mp0;
      BIPOP_MetaParameters _bipop_mp, _bipop_mp0;
      Ran2 ran2;

      void _reset() override
      {
         ran2.seed(-2);
         _mp       = _mp0;
         _bipop_mp = _bipop_mp0;
         _p0       = _parameters;
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