#ifndef NADIR_ABSTRACT_CLASSES_H
#define NADIR_ABSTRACT_CLASSES_H

#include <Eigen/Core>

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

} // namespace nadir
#endif // NADIR_ABSTRACT_CLASSES_H