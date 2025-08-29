#ifndef NADIR_CONTEXT_SIMULATED_ANNEALING_H
#define NADIR_CONTEXT_SIMULATED_ANNEALING_H

#include <Eigen/Core>

/**
 * @file context_simulated_annealing.h
 *
 * Provides a common context structure that all components can use.
 *
 */

namespace nadir
{
/**
 * @brief Context struct for Simulated Annealing
 *
 */
struct SimAnnContext {

      size_t iter;                   //!< Global iteration count
      size_t moves;                  //!< Number of candidate moves tried
      double T;                      //!< Current temperature
      size_t cooling_steps;          //!< Number of cooling steps performed
      size_t failed_candidate_moves; //!< Number of failed candidate moves
      double acceptance_rate;        //!< Current acceptance rate (from historical series)
      double incumbent_sol;          //!< Current incumbent solution
      double best_sol;               //!< Current best solution
      double proposed_sol;           //!< Current proposed solution

      // Global fixed variables, for convenience
      double T0; //!< Initial temperature
      double Tf; //!< Final temperature (may be unused)
};
} // namespace nadir

#endif // NADIR_CONTEXT_SIMULATED_ANNEALING_H