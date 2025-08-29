#ifndef NADIR_ABSTRACT_CLASSES_H
#define NADIR_ABSTRACT_CLASSES_H

#include <Eigen/Core>
#include <optional>
#include <iostream>
#include <sstream>
#include <fstream>

/**
 * @file abstract_classes.h
 *
 * @brief Interfaces for the Cost function and Iteration call back classes
 *
 */

namespace nadir
{
/**
 * @brief Base class for the cost function
 *
 * The user is suppose to derive from this class and provide an implemetnation
 * for its body.
 */
class NadirCostFunction
{
   public:
      /// Virtual destructor
      virtual ~NadirCostFunction() = default;

      /**
       * @brief Main function to evaluate the Cost for Adam minimization, with derivatives
       *
       * @param parameters        Input: array of parameters at the current step
       * @param cost              Output: the location where to write the value of
                                          the Cost function
       * @param cost_derivatives  Output: the array where to write the value of the gradient
                                          of the cost function w.r.t. the parameters
       */
      virtual void Evaluate(const Eigen::VectorXd &parameters, double &cost,
                            Eigen::VectorXd &cost_derivatives)
      {
         (void)parameters;
         (void)cost;
         (void)cost_derivatives;
         throw std::runtime_error(
             "Calling base class implementation of NadirCostFunction. This is a bug.");
      };

      /**
       * @brief Main function to evaluate the Cost for Adam minimization, without derivatives
       *
       * @param parameters        Input: array of parameters at the current step
       * @param cost              Output: the location where to write the value of
                                          the Cost function
       */
      virtual void Evaluate(const Eigen::VectorXd &parameters, double &cost)
      {
         (void)parameters;
         (void)cost;
         throw std::runtime_error(
             "Calling base class implementation of NadirCostFunction. This is a bug.");
      };
};

/**
 * @brief Base class for the Iteration call back function
 *
 * The user is suppose to derive from this class and provide an implemetnation
 * for its body.
 */
class NadirIterCallback
{
   public:
      /// Virtual destructor
      virtual ~NadirIterCallback() = default;

      /**
       * @brief Main function to evaluate the call back
       *
       * @param parameters Input: array of parameters at the current step
       * @return true      To continue the minimization
       * @return false     To abort the minimization
       */
      virtual bool Evaluate(const Eigen::VectorXd &parameters)
      {
         (void)parameters;
         throw std::runtime_error(
             "Calling base class implementation of NadirIterCallback. This is a bug.");
      };
};

/**
 * @brief Base Minimizer class, purely virtual
 *
 * Provides common interface for all derived minimizers
 */
class Minimizer
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
       * @brief Construct a new Adam minimizer
       *
       * @param fnc  The cost function
       * @param pars The array of initial parameters
       */
      Minimizer(NadirCostFunction &fnc, Eigen::VectorXd pars) : _fnc(fnc), _parameters(pars) {};

      void Reset(NadirCostFunction &fnc, Eigen::VectorXd pars)
      {
         std::ostringstream{}.swap(_buffer);
         _fnc_callback.reset();
         _parameters = pars;

         _fnc = fnc;

         _reset();
      }
      /**
       * @brief Empty destructor
       *
       */
      virtual ~Minimizer() = default;
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
      virtual STATUS minimize() = 0;

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
      /// The cost function
      std::reference_wrapper<NadirCostFunction> _fnc;
      /// The (optional) callback
      std::optional<std::reference_wrapper<NadirIterCallback>> _fnc_callback;

      /// Vector of parameters to be updated in the minimization
      Eigen::VectorXd _parameters;

      /// Internal log buffer
      std::ostringstream _buffer;

      virtual void _reset()
      {
      }
};

inline constexpr double pow_n(double x, size_t n)
{
   if (n == 0) return 1.;
   while (n % 2 == 0) {
      n /= 2;
      x *= x;
   }
   double result = x;
   while (n /= 2) {
      x *= x;
      if (n % 2 != 0) result *= x;
   }
   return result;
}

} // namespace nadir
#endif // NADIR_ABSTRACT_CLASSES_H