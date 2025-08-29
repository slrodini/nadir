#ifndef NADIR_BFGS_H
#define NADIR_BFGS_H

#include "nadir/abstract_classes.h"

#include <Eigen/Core>
#include <functional>
#include <deque>

/**
 * @file bfgs.h
 *
 * @brief Class for the Broyden–Fletcher–Goldfarb–Shanno
 * (https://en.wikipedia.org/wiki/Broyden-Fletcher-Goldfarb-Shanno_algorithm)
 *
 */

namespace nadir
{

/**
 * @brief Main BFGS class
 *
 */
class BFGS : public Minimizer
{
   public:
      /// Damping strategy
      enum class DAMPING_TYPE : unsigned {
         NONE   = 0,
         POWELL = 1,
         M_BFGS = 2,
      };

      /// Metaparameters for the Adam minimizer
      struct MetaParameters {
            /// Damping strategy
            DAMPING_TYPE dam_t = BFGS::DAMPING_TYPE::M_BFGS;
            /// Max iterations
            size_t max_it = 1000;
            /// c1
            double c1 = 1.0e-4;
            /// c2
            double c2 = 0.9;
            /// Line search alpha initial
            double alpha = 1.;
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
       * @brief Construct a new BFGS minimizer
       *
       * @param mp   The Meta-parameters
       * @param fnc  The cost function
       * @param pars The array of initial parameters
       */
      BFGS(MetaParameters mp, NadirCostFunction &fnc, Eigen::VectorXd pars);

      virtual ~BFGS() = default;
      ///@}

      void ChangeMetaParameters(MetaParameters mp)
      {
         _mp = mp;
      }

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
      /// Serach direction
      Eigen::VectorXd _p;
      /// Hessian approximation
      Eigen::MatrixXd H;
      /// Inverse Hessian approximation
      Eigen::MatrixXd Hinv;

      /// Trial parameters, used in Wolfe line search
      Eigen::VectorXd _par_trial;
      /// Trial gradient, used in Wolfe line search and then overwritten in main loop
      Eigen::VectorXd _gt_new;

      /// Utility function, shorten the notation
      inline void _cost(const Eigen::VectorXd &p, double &v, Eigen::VectorXd &g)
      {
         _fnc.get().Evaluate(p, v, g);
      }

      double _wolfe_ls(double f_val);

      double _zoom(double f_0, double g_0, double alpha_lo, double f_lo, double g_lo,
                   double alpha_hi, double f_hi, double g_hi);
      double _cu_interpolate(double a0, double f0, double g0, double a1, double f1, double g1);

      void _reset() override;
};

/**
 * @brief Main BFGS class
 *
 */
class LBFGS : public Minimizer
{
   public:
      /// Metaparameters for the Adam minimizer
      struct MetaParameters {
            /// Max iterations
            size_t max_it = 1000;
            /// c1
            double c1 = 1.0e-4;
            /// c2
            double c2 = 0.9;
            /// Memory length for approximate inverse Hessian
            size_t memory = 10;
            /// Line search alpha initial
            double alpha = 1.;
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
       * @brief Construct a new BFGS minimizer
       *
       * @param mp   The Meta-parameters
       * @param fnc  The cost function
       * @param pars The array of initial parameters
       */
      LBFGS(MetaParameters mp, NadirCostFunction &fnc, Eigen::VectorXd pars);

      virtual ~LBFGS() = default;
      ///@}

      void ChangeMetaParameters(MetaParameters mp)
      {
         _mp = mp;
      }

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
      /// Serach direction
      Eigen::VectorXd _p;
      /// History buffer

      /// Trial parameters, used in Wolfe line search
      Eigen::VectorXd _par_trial;
      /// Trial gradient, used in Wolfe line search and then overwritten in main loop
      Eigen::VectorXd _gt_new;

      /// Utility function, shorten the notation
      inline void _cost(const Eigen::VectorXd &p, double &v, Eigen::VectorXd &g)
      {
         _fnc.get().Evaluate(p, v, g);
      }

      /// Determines _p from two loop recursion formula
      void _two_loop_recursion(std::deque<Eigen::VectorXd> &, std::deque<Eigen::VectorXd> &,
                               std::deque<double> &, Eigen::VectorXd &);
      double _wolfe_ls(double f_val);

      double _zoom(double f_0, double g_0, double alpha_lo, double f_lo, double g_lo,
                   double alpha_hi, double f_hi, double g_hi);
      double _cu_interpolate(double a0, double f0, double g0, double a1, double f1, double g1);

      void _reset() override;
};

} // namespace nadir

#endif