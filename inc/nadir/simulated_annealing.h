#ifndef NADIR_SIMULATED_ANNEALING_H
#define NADIR_SIMULATED_ANNEALING_H

#include "nadir/itx_simulated_annealing.h"
#include "nadir/scx_simulated_annealing.h"
#include "nadir/nex_simulated_annealing.h"
#include "nadir/acx_simulated_annealing.h"
#include "nadir/csx_simulated_annealing.h"
#include "nadir/tlx_simulated_annealing.h"

/**
 * @file simulated_annealing.h
 * @brief Interface for the SimAnnealing class
 *
 */
namespace nadir
{
/// Seed the internal random engine. Can be called only once.
void set_seed(long int seed);

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
                   ACx &acx, CSx &csx, TLx &tlx, bool cb_each_step = false)
          : Minimizer(fnc, pars), _incumbent_pars(pars), _current_pars(pars), _itx(itx), _scx(scx),
            _nex(nex), _acx(acx), _csx(csx), _tlx(tlx), _cb_each_step(cb_each_step)
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

         auto [T0, Tf] = _itx();

         double tmp_sol;
         _fnc.get().Evaluate(_parameters, tmp_sol);

         context = SimAnnContext{
             .iter                   = 0,
             .moves                  = 0,
             .T                      = T0,
             .cooling_steps          = 0,
             .failed_candidate_moves = 0,
             .acceptance_rate        = 1,
             .incumbent_sol          = tmp_sol,
             .best_sol               = tmp_sol,
             .proposed_sol           = tmp_sol,
             .T0                     = T0,
             .Tf                     = Tf,
         };
      };

      /**
       * @brief Construct a new SimAnnealing Minimizer
       *
       * @param prev_context  Context from a previous run of SimAnnealing
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
      SimAnnealing(SimAnnContext prev_context, NadirCostFunction &fnc, Eigen::VectorXd pars,
                   ITx &itx, SCx &scx, NEx &nex, ACx &acx, CSx &csx, TLx &tlx,
                   bool cb_each_step = false)
          : Minimizer(fnc, pars), _incumbent_pars(pars), _current_pars(pars), _itx(itx), _scx(scx),
            _nex(nex), _acx(acx), _csx(csx), _tlx(tlx), _cb_each_step(cb_each_step)
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

         context = prev_context;
      };

      /// Main minimize function
      STATUS minimize() override;

      /// Set the inital parameters
      void SetInitialParameters(const Eigen::VectorXd &pars) override
      {
         _parameters     = pars;
         _incumbent_pars = pars;
         _current_pars   = pars;
      }

      /// Set the i-th inital parameter
      void SetInitialParameter(long int index, double par) override
      {
         _parameters(index)     = par;
         _incumbent_pars(index) = par;
         _current_pars(index)   = par;
      }

      SimAnnContext GetContext() const
      {
         return context;
      }

   private:
      Eigen::VectorXd
          _incumbent_pars; //!< Local storage for the parameters of the incumbent solution
      Eigen::VectorXd _current_pars; //!< Local storage for the parameters of the proposed solution

      ITx &_itx; //!< Initial and Final temperature decider
      SCx &_scx; // !< Stopping criteria
      NEx &_nex; // !< Neighbour determination
      ACx &_acx; // !< Acceptance criteria
      CSx &_csx; // !< Cooling scheme
      TLx &_tlx; // !< Temperature length scheme

      SimAnnContext context; //!< The context for the SA
      bool _cb_each_step; //!< Whether to call the IterCallBack at each step, or just on the update
                          //!< of best parameters
};

} // namespace nadir
#endif // NADIR_SIMULATED_ANNEALING_H