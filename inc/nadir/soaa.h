#ifndef NADIR_SOAA_H
#define NADIR_SOAA_H

#include "nadir/abstract_classes.h"

#include <Eigen/Core>
#include <functional>

/**
 * @file soaa.h
 *
 * @brief Class for SOAA algorithm https://arxiv.org/pdf/2410.02293
 *
 */

namespace nadir
{

/**
 * @brief Main SOAA class
 *
 * Although similar to Adam and inspired by it, to this algorithm I dedicate a separate class to
 * have minimal codependence from the main Adam variants
 */
class SOAA : public Minimizer
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
            /// Parameter for the decoupled weight decay variants
            double gamma = 0.1;
            /// Parameter for the decoupled weight decay variants
            double lambda = 0.;
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
       * @brief Construct a new SOAA minimizer
       *
       * @param fnc  The cost function
       * @param pars The array of initial parameters
       */
      SOAA(const MetaParameters &mp, NadirCostFunction &fnc, Eigen::VectorXd pars);

      /**
       * @brief Construct a new SOAA minimizer
       *
       * @param fnc   The cost function
       * @param n_par The number of parameters (initial parameters set to 0)
       */
      SOAA(const MetaParameters &mp, NadirCostFunction &fnc, long int n_par = 1);

      virtual ~SOAA() = default;
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
      /// Values of previous and current cost function
      double f_new, f_old;

      /// Internal temporary vectors for the minimizer
      Eigen::VectorXd _gt, _mt, _vt;

      /// Additional temporary vector for some of the Adam variants
      Eigen::VectorXd _mt_hat, _vt_hat, _Ft;

      /// Trust region scaling
      double _dt;
      /// Loss average
      double _l_ave;
      /// Predicted reduction
      double _pr;

      /// The meta-parameters of the minimizer
      MetaParameters _mp;

      /// The scheduler for a time-dependent learning rate (defaulted to unit function)
      std::function<double(size_t)> _scheduler;

      /// Sign function, for internal use only
      const std::function<double(double)> sign = [](double x) {
         return (x > 0) - (x < 0);
      };
};
} // namespace nadir

#endif