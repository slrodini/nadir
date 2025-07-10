#ifndef NADIR_ADAM_H
#define NADIR_ADAM_H

#include "nadir/abstract_classes.h"

#include <Eigen/Core>
#include <functional>
#include <optional>
#include <iostream>
#include <sstream>
#include <fstream>

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
class Adam
{
   public:
      /// Status on exit of the minimizer
      enum class STATUS : unsigned {
         SUCCESS  = 0,
         ABORT    = 1,
         MAX_IT   = 2,
         LOW_DIFF = 3,
         CONTINUE = 4,
      };

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
      Adam(NadirCostFunction &fnc, Eigen::VectorXd pars = Eigen::VectorXd::Zero(1));

      /**
       * @brief Construct a new Adam minimizer
       *
       * @param fnc   The cost function
       * @param n_par The number of parameters (initial parameters set to 0)
       */
      Adam(NadirCostFunction &fnc, long int n_par = 1);
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
       * @brief Set the initial parameters
       *
       * @param pars The array of initial parameters
       */
      void SetInitialParameters(const Eigen::VectorXd &pars)
      {
         _parameters = pars;
      }

      /**
       * @brief Add a callback to the minimizer
       *
       * @param f The call back
       */
      void AddCallBack(NadirIterCallback &f)
      {
         _fnc_callback = f;
      }

      /// Return the current set of parameters
      const Eigen::VectorXd &GetParameters() const
      {
         return _parameters;
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
      STATUS minimize();

      /// Flus internal lig buffer to file
      void FlusToFile(const std::string &filename) const
      {
         std::ofstream file(filename);
         if (file.is_open()) {
            file << _buffer.str();
            file.close();
         } else {
            throw std::runtime_error("FlushToFile: Could not open file: " + filename);
         }
      }

      /// Flus internal lig buffer to the stdout
      void FlusToStdout() const
      {
         std::cout << _buffer.str() << std::endl;
      }

   private:
      /// The cost function
      std::reference_wrapper<NadirCostFunction> _fnc;
      /// The (optional) callback
      std::optional<std::reference_wrapper<NadirIterCallback>> _fnc_callback;

      /// Vector of parameters to be updated in the minimization
      Eigen::VectorXd _parameters;

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

      /// Internal log buffer
      std::ostringstream _buffer;

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