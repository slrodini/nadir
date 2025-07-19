#ifndef NADIR_LEVENBERG_MARQUARDT_H
#define NADIR_LEVENBERG_MARQUARDT_H

#include <Eigen/Dense>
#include "nadir/abstract_classes.h"

namespace nadir
{
class NadirLevMarCostFunction
{
   public:
      virtual ~NadirLevMarCostFunction() = default;

      virtual void Evaluate(const Eigen::VectorXd &parameters, Eigen::VectorXd &residual,
                            Eigen::Block<Eigen::MatrixXd> *jacobian) = 0;

      virtual long int ResidualNumber() = 0;
};

/**
 * @brief Minimizer class, for Levenberg-Marquardt algorithm
 *
 * The implementation is by no mean refined in terms of storage used or optimization
 */
class LevMarMinimizer
{
   public:
      /// Status on exit of the minimizer
      enum class STATUS : unsigned {
         SUCCESS  = 0,
         ABORT    = 1,
         MAX_IT   = 2,
         LOW_DIFF = 3,
         CONTINUE = 4,
         FAILURE  = 5,
      };

      struct MetaParameters {
            /// Max iter
            size_t max_iterations = 1000;
            /// Tollerance on the gradient norm
            double grad_toll = 1.0e-8;
            /// Tollerance on the \f$ \Delta F \f$  in subsequent steps
            double diff_value_toll = 1.0e-8;
            /// Initial trust region radius
            double mu = 1.0e4;
            /// Min threshold to accept step
            double eps = 1.0e-3;
            /// Lower bound to increase trust region readius
            double eta1 = 0.9;
            /// Upper bound to decrease trust region radius
            double eta2 = 0.1;
            /// Real time progress to stderr
            bool real_time_progress = false;
            /// Function to return 1/sqrt(mu) or variants
            std::function<double(double)> scaling_function = [](double t) {
               return 1.0 / sqrt(t);
            };
      };

      /// Print the status as a string
      static inline std::string print_status(STATUS s)
      {
         switch (s) {
         case STATUS::SUCCESS:
            return "SUCCESS";
            break;
         case STATUS::ABORT:
            return "ABORT";
            break;
         case STATUS::MAX_IT:
            return "MAX_IT";
            break;
         case STATUS::LOW_DIFF:
            return "LOW_DIFF";
            break;
         case STATUS::CONTINUE:
            return "CONTINUE";
            break;
         case STATUS::FAILURE:
            return "FAILURE";
            break;
         default:
            return "";
         }
      }

      /// \name Constructor
      ///@{

      /**
       * @brief Construct a new minimizer
       *
       * @param fnc  The cost function
       * @param pars The array of initial parameters
       */
      LevMarMinimizer(MetaParameters &mp, NadirLevMarCostFunction &fnc, Eigen::VectorXd pars);

      /**
       * @brief Empty destructor
       *
       */
      virtual ~LevMarMinimizer() = default;
      ///@}

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

   protected:
      /// The meta-parameters
      MetaParameters _mp;
      /// The cost function
      std::reference_wrapper<NadirLevMarCostFunction> _fnc;
      /// The (optional) callback
      std::optional<std::reference_wrapper<NadirIterCallback>> _fnc_callback;

      /// Vector of parameters to be updated in the minimization
      Eigen::VectorXd _parameters;

      /// Internal log buffer
      std::ostringstream _buffer;

      /// Local storage
      Eigen::VectorXd residual, residual_new;
      Eigen::MatrixXd Jacobian;
      double f_old, f_new;

      /// For convenience
      long int npar, nres;
};

} // namespace nadir

#endif