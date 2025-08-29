#ifndef NADIR_DIFFERENTIAL_EVOLUTION_H
#define NADIR_DIFFERENTIAL_EVOLUTION_H

#include "nadir/abstract_classes.h"
#include "nadir/ran2.h"
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
      };

      void ChangeMetaParameters(MetaParameters mp)
      {
         _mp = mp;
      }

      /**
       * @brief Main function: execute the minimization
       *
       * @return STATUS, either SUCCESS or ABORT
       */
      STATUS minimize() override;

   private:
      MetaParameters _mp; //!< Meta-parameters
      Ran2 ran2;
      /// Check consistency of metaparameters
      void _check_meta_parameters();
      /// Select three distinct indexes from each other and from fix index x
      void _get_a_b_c_index(size_t x, size_t &a, size_t &b, size_t &c);
      /// Extract best set of parameters from population
      size_t _get_best_parameters(std::vector<Eigen::VectorXd> *pop, std::vector<double> *costs);

      void _reset() override
      {
         ran2.seed(-2);
      }
};

class jSOa : public Minimizer
{
   public:
      /// Differential Evolution meta-parameters
      struct MetaParameters {
            /// initial Population size >= 4
            size_t NP_ini = 100;
            /// Minimal population size
            size_t NP_min = 4;
            /// Highest p-percentile
            double p_max = 0.25;
            /// Lowest p-percentile
            double p_min = 0.125;
            /// Use 'a' variant of jSO
            bool impr_archive_eviction = true;
            /// If jSOa, percentage of tail
            double ap = 0.2;
            /// Terminal value for CR
            double CR_term = 0.1;
            /// History size
            size_t H = 5;
            /// Search width
            double width = 1.;
            /// Max iterations
            size_t max_iter = 100;
            /// Progress to std::cout(cerr)
            bool real_time_progress = false;
      };

      /**
       * @brief Construct a new DiffEvolution Minimizer
       *
       * @param mp   Meta-parameters
       * @param fnc  Cost function
       * @param pars Initial parameters (center of random generation)
       */
      jSOa(MetaParameters mp, NadirCostFunction &fnc, Eigen::VectorXd pars)
          : Minimizer(fnc, pars), _mp(mp)
      {
         _check_meta_parameters();
      };

      /**
       * @brief Main function: execute the minimization
       *
       * @return STATUS, either SUCCESS or ABORT
       */
      STATUS minimize() override;

   private:
      MetaParameters _mp; //!< Meta-parameters
      Ran2 ran2;

      /// Extract best set of parameters from population
      size_t _get_best_parameters(std::vector<Eigen::VectorXd> *pop, std::vector<double> *costs);
      void _check_meta_parameters();

      void _reset() override
      {
         ran2.seed(-2);
      }
};

} // namespace nadir
#endif // NADIR_DIFFERENTIAL_EVOLUTION_H