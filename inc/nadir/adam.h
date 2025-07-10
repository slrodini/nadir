#ifndef NADIR_ADAM_H
#define NADIR_ADAM_H

#include "nadir/abstract_classes.h"

#include <Eigen/Core>
#include <functional>
#include <optional>

/**
 * @file adam.h
 *
 * @brief Class for family of Adam adjecent algorithms
 *
 */

namespace nadir
{

/**
 * @brief Main Adam class
 *
 * Adam minimizer class. It stores the parameters, references to the Cost function and iteration
 * callbacks, and some internal variables. The main function is `minimize` to perform the
 * minimization.
 */
class Adam : public Minimizer
{
   public:
      /// Variant of the Adam base algorithm
      enum class ADAM_VARIANT : unsigned {
         CLASSIC,
         AMSGRAD,
         AMSGRAD_V2,
         NADAM,
         ADAMW,
         ADABELIEF,
         ADABELIEF_W, // AdaBelief with decoupled weight decay
         LION,
         RADAM,
      };

      /// Metaparameters for the Adam minimizer
      struct MetaParameters {
            /// Adam varian
            ADAM_VARIANT variant = ADAM_VARIANT::CLASSIC;
            /// Max number of iteration
            size_t max_it = 100;
            /// Learning rate
            double alpha = 0.001;
            /// Exponentaial decay rate for the moving average of the 1st moment
            double beta1 = 0.9;
            /// Exponentaial decay rate for the moving average of the 2st moment
            double beta2 = 0.999;
            /// Regulator of the denominator in the update
            double eps = 1.0e-8;
            /// Tollerance on the gradient norm
            double grad_toll = 1.0e-8;
            /// Tollerance on the \f$ \Delta F \f$  in subsequent steps
            double diff_value_toll = 1.0e-8;
            /// Parameter for the decoupled weight decay variants
            double lambda = 0.;
      };

      /// \name Constructor
      ///@{

      /**
       * @brief Construct a new Adam minimizer
       *
       * @param fnc  The cost function
       * @param pars The array of initial parameters
       */
      Adam(NadirCostFunction &fnc, Eigen::VectorXd pars);

      /**
       * @brief Construct a new Adam minimizer
       *
       * @param fnc   The cost function
       * @param n_par The number of parameters (initial parameters set to 0)
       */
      Adam(NadirCostFunction &fnc, long int n_par = 1);

      virtual ~Adam() = default;
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

      /**
       * @brief Set the meta-parameters for the minimizer
       *
       * @param mp The meta-parameters
       *
       * \note If this function is not called, the meta-parameters are setted to the default ones
       * which may be not appropriate for the problem
       */
      void SetMetaParameters(const MetaParameters &mp)
      {
         _mp = mp;
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
      /// Values of previous and current cost function
      double f_new, f_old;

      /// Internal temporary vectors for the minimizer
      Eigen::VectorXd _gt, _mt, _vt;

      /// Additional temporary vector for some of the Adam variants
      Eigen::VectorXd _vt_hat;

      /// The meta-parameters of the minimizer
      MetaParameters _mp;

      /// The scheduler for a time-dependent learning rate (defaulted to unit function)
      std::function<double(size_t)> _scheduler;

      /// \name Step functions
      ///@{
      size_t step(size_t);
      size_t step_ams(size_t);
      size_t step_ams_v2(size_t);
      size_t step_nadam(size_t);
      size_t step_adamw(size_t);
      size_t step_adabelief(size_t);
      size_t step_adabelief_w(size_t);
      size_t step_lion(size_t);
      size_t step_radam(size_t);
      ///@}

      /// Sign function, for internal use only
      const std::function<double(double)> sign = [](double x) {
         return (x > 0) - (x < 0);
      };
};
} // namespace nadir

#endif