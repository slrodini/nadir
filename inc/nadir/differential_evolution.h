#ifndef NADIR_DIFFERENTIAL_EVOLUTION_H
#define NADIR_DIFFERENTIAL_EVOLUTION_H

#include "nadir/abstract_classes.h"

#include <Eigen/Core>

/**
 * @file differential_evolution.h
 *
 * @brief Class for Differential Evolution minimizer
 *
 * Implementation is taken from https://en.wikipedia.org/wiki/Differential_evolution
 */

namespace nadir
{
class DiffEvolution : public Minimizer
{
   public:
      /// Differential Evolution meta-parameters
      struct MetaParameters {
            size_t NP       = 4;   //!< Population size >= 4
            double CR       = 0.9; //!< Cross-over probability in (0, 1)
            double F        = 0.8; //!< Differential weight in [0, 2]
            double width    = 1.;  //!< Std for the Gaussian initialization of initial population
            size_t max_iter = 100; //!< Max number of iterations
            bool real_time_progress = false; //!< Real time stdout
      };

      /**
       * @brief Construct a new DiffEvolution Minimizer
       *
       * @param mp   Meta-parameters
       * @param fnc  Cost function
       * @param pars Initial parameters (center of random generation)
       */
      DiffEvolution(MetaParameters mp, NadirCostFunction &fnc, Eigen::VectorXd pars)
          : Minimizer(fnc, pars), _mp(mp)
      {
         _check_meta_parameters();
         _init();
      };

      /**
       * @brief Construct a new DiffEvolution Minimizer
       *
       * @param mp    Meta-parameters
       * @param fnc   Cost function
       * @param n_par Number of parameters
       */
      DiffEvolution(MetaParameters mp, NadirCostFunction &fnc, long int n_par = 1)
          : Minimizer(fnc, n_par), _mp(mp)
      {
         _check_meta_parameters();
         _init();
      }

      virtual void SetInitialParameters(const Eigen::VectorXd &pars) override
      {
         _parameters = pars;
         _init();
      }

      /// Remove ability to set parameters individually after initialization
      virtual void SetInitialParameter(long int, double) override
      {
         throw std::runtime_error(
             "Differential Evolution cannot set parameters after initialization.");
      }

      /**
       * @brief Main function: execute the minimization
       *
       * @return STATUS, either SUCCESS or ABORT
       */
      STATUS minimize() override;

   private:
      MetaParameters _mp;                           //!< Meta-parameters
      std::vector<Eigen::VectorXd> _population_old; //!< Population of parameters
      std::vector<Eigen::VectorXd> _population_new; //!< Population of parameters
      std::vector<double> _costs_old; //!< Values of cost functions per parameter set in population
      std::vector<double> _costs_new; //!< Values of cost functions per parameter set in population

      /// Check consistency of metaparameters
      void _check_meta_parameters();
      /// Reset the Minimizer state
      void _init();
      /// Select three distinct indexes from each other and from fix index x
      void _get_a_b_c_index(size_t x, size_t &a, size_t &b, size_t &c);
      /// Extract best set of parameters from population
      size_t _get_best_parameters(std::vector<Eigen::VectorXd> *pop, std::vector<double> *costs);
};

} // namespace nadir
#endif // NADIR_DIFFERENTIAL_EVOLUTION_H