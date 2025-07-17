#ifndef NADIR_TADAM_H
#define NADIR_TADAM_H

#include "nadir/abstract_classes.h"

#include <Eigen/Core>
#include <functional>

/**
 * @file tadam.h
 *
 * @brief Class for TAdam algorithm https://doi.org/10.1016/j.neunet.2023.09.010
 *
 */

namespace nadir
{

/**
 * @brief Main TAdam class
 *
 * Although similar to Adam and inspired by it, to this algorithm I dedicate a separate class to
 * have minimal codependence from the main Adam variants
 */
class TAdam : public Minimizer
{
   public:
      /// Metaparameters for the Adam minimizer
      struct MetaParameters {
            /// Max number of iteration
            size_t max_it = 100;
            /// Learning rate
            double alpha = 0.001;
            /// Exponentaial decay rate for the moving average of the 1st moment
            double beta1 = 0.9;
            /// Exponentaial decay rate for the moving average of the 2st moment
            double beta2 = 0.999;
            /// Trust region radius
            double delta_0 = 10.;
            /// Algorithm performance benchmark
            double gamma = 0.;
            /// Regulator of the denominator in the update
            double eps = 1.0e-8;
            /// Tollerance on the gradient norm
            double grad_toll = 1.0e-8;
            /// Tollerance on the \f$ \Delta F \f$  in subsequent steps
            double diff_value_toll = 1.0e-8;
      };

      /// \name Constructor
      ///@{

      /**
       * @brief Construct a new TAdam minimizer
       *
       * @param fnc  The cost function
       * @param pars The array of initial parameters
       */
      TAdam(const MetaParameters &mp, NadirCostFunction &fnc, Eigen::VectorXd pars);

      /**
       * @brief Construct a new TAdam minimizer
       *
       * @param fnc   The cost function
       * @param n_par The number of parameters (initial parameters set to 0)
       */
      TAdam(const MetaParameters &mp, NadirCostFunction &fnc, long int n_par = 1);

      virtual ~TAdam() = default;
      ///@}

      /**
       * @brief Set the Scheduler object for variable learning rate
       *
       * @param scheduler The sceduler, as a function of the step count
       */
      void SetScheduler(const std::function<double(size_t)> &scheduler)
      {
         _scheduler = scheduler;
      }

      /// Returns the meta-parameters
      const MetaParameters &GetMetaParameters() const
      {
         return _mp;
      }

      /**
       * @brief Main function: execute the minimization
       *
       * @return STATUS
       */
      STATUS minimize() override;

   private:
      /// The meta-parameters of the minimizer
      MetaParameters _mp;

      /// The scheduler for a time-dependent learning rate (defaulted to unit function)
      std::function<double(size_t)> _scheduler;
};
} // namespace nadir

#endif