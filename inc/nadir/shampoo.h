#ifndef NADIR_SHAMPOO_H
#define NADIR_SHAMPOO_H

#include "nadir/abstract_classes.h"

#include <Eigen/Core>
#include <functional>

/**
 * @file shampoo.h
 *
 * @brief Class for the Shampoo algorithm https://arxiv.org/pdf/1802.09568
 *
 */

namespace nadir
{

/**
 * @brief Main Shampoo class
 *
 */
class Shampoo : public Minimizer
{
   public:
      /// Metaparameters for the Adam minimizer
      struct MetaParameters {
            /// Max iterations
            size_t max_it = 1000;
            /// # of steps between updates of G
            double beta = 0.9;
            /// Learning rate
            double lambda = 0.01;
            /// Exponent parameter
            double p = 1.;
            /// Regulator
            double eps = 1.0e-8;
            /// Tollerance on the gradient norm
            double grad_toll = 1.0e-8;
            /// Tollerance on the \f$ \Delta F \f$  in subsequent steps
            double diff_value_toll = 1.0e-8;
      };

      /// \name Constructor
      ///@{

      /**
       * @brief Construct a new Shampoo minimizer
       *
       * @param mp   The Meta-parameters
       * @param fnc  The cost function
       * @param pars The array of initial parameters
       */
      Shampoo(MetaParameters &mp, NadirCostFunction &fnc, Eigen::VectorXd pars);

      /**
       * @brief Construct a new Shampoo minimizer
       *
       * @param mp   The Meta-parameters
       * @param fnc   The cost function
       * @param n_par The number of parameters (initial parameters set to 0)
       */
      Shampoo(MetaParameters &mp, NadirCostFunction &fnc, long int n_par = 1);

      virtual ~Shampoo() = default;
      ///@}

      /**
       * @brief Main function: execute the minimization
       *
       * @return STATUS
       */
      STATUS minimize() override;

   private:
      /// Meta-parameters
      MetaParameters _mp;
      /// Gradient
      Eigen::VectorXd _gt;
      Eigen::MatrixXd _G_acc;
      Eigen::MatrixXd _P;

      /// epsilon * I
      Eigen::MatrixXd _eps_Id;
};
} // namespace nadir

#endif