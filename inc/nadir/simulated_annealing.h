#ifndef NADIR_SIMULATED_ANNEALING_H
#define NADIR_SIMULATED_ANNEALING_H

#include "nadir/itx_simulated_annealing.h"
#include "nadir/scx_simulated_annealing.h"
#include "nadir/nex_simulated_annealing.h"
#include "nadir/acx_simulated_annealing.h"
#include "nadir/csx_simulated_annealing.h"
#include "nadir/tlx_simulated_annealing.h"

// Note: Temperature restart is not contemplated in this implementation
// because one can chain Minimizers, therefore there is no need to
// restart the temperature inside the annealing.
// If some meta-parameters and components want to have memory of what
// happened in previous Minimizers, this has to be chained by the user.

namespace nadir
{
void set_seed(long int seed);

class SimAnnealing : public Minimizer
{
   public:
      SimAnnealing(NadirCostFunction &fnc, Eigen::VectorXd pars, ITx &itx, SCx &scx, NEx &nex,
                   ACx &acx, CSx &csx, TLx &tlx)
          : Minimizer(fnc, pars), _incumbent_pars(pars), _current_pars(pars), _itx(itx), _scx(scx),
            _nex(nex), _acx(acx), _csx(csx), _tlx(tlx)
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

      STATUS minimize() override;

      void SetInitialParameters(const Eigen::VectorXd &pars) override
      {
         _parameters     = pars;
         _incumbent_pars = pars;
         _current_pars   = pars;
      }
      void SetInitialParameter(long int index, double par) override
      {
         _parameters(index)     = par;
         _incumbent_pars(index) = par;
         _current_pars(index)   = par;
      }

   private:
      Eigen::VectorXd _incumbent_pars;
      Eigen::VectorXd _current_pars;

      ITx &_itx; //!< Initial and Final temperature decider
      SCx &_scx; // !< Stopping criteria
      NEx &_nex; // !< Neighbour determination
      ACx &_acx; // !< Acceptance criteria
      CSx &_csx; // !< Cooling scheme
      TLx &_tlx; // !< Temperature length scheme
};

} // namespace nadir
#endif // NADIR_SIMULATED_ANNEALING_H