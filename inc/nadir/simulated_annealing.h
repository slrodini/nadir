#ifndef NADIR_SIMULATED_ANNEALING_H
#define NADIR_SIMULATED_ANNEALING_H

#include "nadir/sim_ann/itx_simulated_annealing.h"
#include "nadir/sim_ann/scx_simulated_annealing.h"
#include "nadir/sim_ann/nex_simulated_annealing.h"
#include "nadir/sim_ann/acx_simulated_annealing.h"
#include "nadir/sim_ann/csx_simulated_annealing.h"
#include "nadir/sim_ann/tlx_simulated_annealing.h"

// TODO: rework the interface for this!

/**
 * @file simulated_annealing.h
 * @brief Interface for the SimAnnealing class
 *
 */
namespace nadir
{

/// Simulated Annealing Minimizer
class SimAnnealing : public Minimizer
{
   public:
      /**
       * @brief Construct a new SimAnnealing Minimizer
       *
       * @param fnc  The cost function
       * @param pars The initial set of parameters
       * @param itx  The ITx component (see README)
       * @param scx  The SCx component (see README)
       * @param nex  The NEx component (see README)
       * @param acx  The ACx component (see README)
       * @param csx  The CSx component (see README)
       * @param tlx  The TLx component (see README)
       * @param cb_each_step Whether to call the NadirIterCallback at each step, or just successful
       * ones
       */
      SimAnnealing(NadirCostFunction &fnc, Eigen::VectorXd pars, ITx &itx, SCx &scx, NEx &nex,
                   ACx &acx, CSx &csx, TLx &tlx, bool real_time_progress = false,
                   bool cb_each_step = false)
          : Minimizer(fnc, pars), _incumbent_pars(pars), _current_pars(pars), _itx(itx), _scx(scx),
            _nex(nex), _acx(acx), _csx(csx), _tlx(tlx), _cb_each_step(cb_each_step),
            _real_time_progress(real_time_progress)
      {

         if (_scx.tag == SCxCase::SC3) {
            if (_csx.tag == CSxCase::CS1 || _csx.tag == CSxCase::CS9 ||
                _csx.tag == CSxCase::CS10 || _csx.tag == CSxCase::CS11) {
               throw std::invalid_argument(
                   "Incompatible stopping condition (SC3), with the given Cooling Schedule. The "
                   "selected cooling schedule is not guaranteed to produce arbitrarly small "
                   "temperature values for sufficiently large times. An iteration-count based "
                   "stopping criteria should be used. ");
            }
         }
      };

      /// Main minimize function
      STATUS minimize() override;

      SimAnnContext GetContext() const
      {
         return context;
      }

   private:
      Eigen::VectorXd
          _incumbent_pars; //!< Local storage for the parameters of the incumbent solution
      Eigen::VectorXd _current_pars; //!< Local storage for the parameters of the proposed solution

      // Ran2 ran2;

      ITx &_itx; //!< Initial and Final temperature decider
      SCx &_scx; // !< Stopping criteria
      NEx &_nex; // !< Neighbour determination
      ACx &_acx; // !< Acceptance criteria
      CSx &_csx; // !< Cooling scheme
      TLx &_tlx; // !< Temperature length scheme

      SimAnnContext context; //!< The context for the SA
      /// Whether to call the IterCallBack at each step, or just on the update of best parameters
      bool _cb_each_step;
      bool _real_time_progress;

      void _reset() override;
};

} // namespace nadir
#endif // NADIR_SIMULATED_ANNEALING_H