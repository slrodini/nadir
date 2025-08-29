#ifndef NADIR_ITX_SIMULATED_ANNEALING_H
#define NADIR_ITX_SIMULATED_ANNEALING_H

#include "nadir/abstract_classes.h"

/**
 * @file itx_simulated_annealing.h
 *
 * ITx (Initial temperature) components for the simulated
 * annealing, as presented in Computers and Operations Research 104 (2019) 191–206
 * (https://doi.org/10.1016/j.cor.2018.12.015)
 *
 */

namespace nadir
{

/// Pre-implemented components
enum class ITxCase {
   IT1, //!< fixed initial temperature T0 = k
   IT2, //!< T0 proportional to initial cost T0 = k C(0)
   IT3, //!< T0 proportional to max gap in consecutive costs in a random walk
   IT4, //!< T0 proportional to average gap in consecutive costs in a random walk
   IT6, //!< T0 as fixed probability p0 in a random walk
   IT7, //!< T0 from random walk, using min, max and average of gaps
};

/// General interface for ITx
class ITx
{
   public:
      const ITxCase tag; //!< Which ITx are we?

      /// For RTTI
      virtual ~ITx() = default;

      /// Caller
      virtual std::pair<double, double> operator()() = 0;

   protected:
      /// Protected base constructor, force tag specification from child classes
      ITx(ITxCase which) : tag(which)
      {
      }
};

/// Constant T0
class IT1 : public ITx
{
   public:
      /**
       * @brief Constructor
       *
       * @param k  Initial temperature (in the notation of the paper)
       * @param Tf Final temperature, user defined
       */
      IT1(double k, double Tf) : ITx(ITxCase::IT1), _k(k), _Tf(Tf) {};
      /// Caller
      std::pair<double, double> operator()() override;

   private:
      ///\name Cached variables
      ///@{
      double _k, _Tf;
      ///@}
};

/// Constant T0, proportional to initial Cost
class IT2 : public ITx
{
   public:
      /**
       * @brief Constructor, initial temperature is \f$ k C(p_0) \f$
       *
       * @param k   Initial temperature multiplier
       * @param Tf  Final temperature (user defined)
       * @param fnc Cost function
       * @param par Initial parameters
       */
      IT2(double k, double Tf, NadirCostFunction &fnc, Eigen::VectorXd &par)
          : ITx(ITxCase::IT2), _k(k), _Tf(Tf), _fnc(fnc), _par(par) {};

      std::pair<double, double> operator()() override;

   private:
      /// \name Cached variables
      ///@{
      double _k, _Tf;
      NadirCostFunction &_fnc;
      Eigen::VectorXd &_par;
      ///@}
};

/// Implements random walk for gapped initial temperature
void itx_random_walk(NadirCostFunction &fnc, Eigen::VectorXd &par, size_t n, double step_size,
                     std::vector<double> &out);

/// Min-Max random walk
class IT3 : public ITx
{
   public:
      /**
       * @brief Constructor
       *
       * @param k         Temperature multiplier
       * @param n         Random walk steps
       * @param step_size Random walk step size
       * @param fnc       Cost function
       * @param par       Initial parameters
       *
       * The temperature are defined by
       * \f[ T_0 = k\  \text{max}_{1\le i\le n} |\Delta_{i, i+1}|\f],
       * \f[ T_f = k\  \text{min}_{1\le i\le n} |\Delta_{i, i+1}|\f]
       * where \f$ \Delta_{i, i+1} \f$ is the Cost difference between two adjecent random walk
       * steps.
       */
      IT3(double k, size_t n, double step_size, NadirCostFunction &fnc, Eigen::VectorXd &par)
          : ITx(ITxCase::IT2), _k(k), _n(n), _step_size(step_size), _fnc(fnc), _par(par) {};

      /// Caller
      std::pair<double, double> operator()() override;

   private:
      /// \name Cached variables
      ///@{
      double _k;
      size_t _n;
      double _step_size;
      NadirCostFunction &_fnc;
      Eigen::VectorXd _par;
      ///@}
};

/// Average random walk
class IT4 : public ITx
{
   public:
      /**
       * @brief Constructor
       *
       * @param k  Initial temperature multiplier
       * @param Tf Final temperature (user defined)
       * @param n  Random walk steps
       * @param step_size Random walk step size
       * @param fnc Cost function
       * @param par Inital parameters
       *
       * The inital temperature is define by
       * \f[ T_0 = \frac{k}{n}\  \sum_{i=1}^n-1  |\Delta_{i, i+1}|\f],
       * where \f$ \Delta_{i, i+1} \f$ is the Cost difference between two adjecent random walk
       * steps.
       */
      IT4(double k, double Tf, size_t n, double step_size, NadirCostFunction &fnc,
          Eigen::VectorXd &par)
          : ITx(ITxCase::IT2), _k(k), _Tf(Tf), _n(n), _step_size(step_size), _fnc(fnc),
            _par(par) {};

      /// Caller
      std::pair<double, double> operator()() override;

   private:
      /// \name Cached variables
      ///@{
      double _k, _Tf;
      size_t _n;
      double _step_size;
      NadirCostFunction &_fnc;
      Eigen::VectorXd _par;
      ///@}
};

/// Average probability random walk
class IT6 : public ITx
{
   public:
      /**
       * @brief Constructor
       *
       * @param k  Inital temperature multiplier
       * @param Tf Final temperature (user defined)
       * @param p0 Probability
       * @param n  Random walk steps
       * @param step_size Random walk step size
       * @param fnc Cost function
       * @param par Inital parameters
       * @throw invalid_argument if p0 is not in (0, 1)
       *
       * The initial temperature is defined by
       * \f[ T_0 = |k\ \Delta_{av} / \log(p_0)| \f]
       * where \f$ p_0 \in (0, 1) \f$ is a fixed initial probaility and
       * \f[ \Delta_{av} = \frac{2}{n(n-1)} \sum_{i=1}^n \sum_{j=i+1}^n |\Delta_{ij}|\f]
       * and \f$ \Delta_{ij} \f$ is the cost function differnce between the step i and step j
       * of the random walk.
       */
      IT6(double k, double Tf, double p0, size_t n, double step_size, NadirCostFunction &fnc,
          Eigen::VectorXd &par)
          : ITx(ITxCase::IT2), _k(k), _Tf(Tf), _p0(p0), _n(n), _step_size(step_size), _fnc(fnc),
            _par(par)
      {
         if (_p0 <= 0 || _p0 >= 1) throw std::invalid_argument("IT6: p0 must be in (0, 1)");
      };

      std::pair<double, double> operator()() override;

   private:
      /// \name Cached variables
      ///@{
      double _k, _Tf;
      double _p0;
      size_t _n;
      double _step_size;
      NadirCostFunction &_fnc;
      Eigen::VectorXd _par;
      ///@}
};

/// Min-Max-Average random walk
class IT7 : public ITx
{
   public:
      /**
       * @brief Constructor
       *
       * @param k  Temperature multiplier
       * @param l1  \f$ \lambda_1\f$ parameter
       * @param l1p \f$ \lambda_1'\f$ parameter
       * @param l2  \f$ \lambda_2\f$ parameter
       * @param l2p \f$ \lambda_2'\f$ parameter
       * @param n  Random walk steps
       * @param step_size Random walk step size
       * @param fnc Cost function
       * @param par Inital parameters
       * @throws invalid_argument if the \f$ \lambda \f$ parameters are not in [0, 1] and if
       * \f$ \lambda_i + \lambda_i' > 1 \f$
       *
       * The initial temperature and final temperature are defined by
       * \f[ T_0 = k \left[ (1-\lambda_1-\lambda_1')\Delta_{min} + \lambda_1 \Delta_{av} +
       * \lambda_1' \Delta_{max} \right] \f]
       * \f[ T_f = k \left[ (1-\lambda_2-\lambda_2')\Delta_{min} + \lambda_2 \Delta_{av} +
       * \lambda_2' \Delta_{max} \right] \f]
       * where
       * \f[ \Delta_{av}  = \frac{2}{n(n-1)} \sum_{i=1}^n \sum_{j=i+1}^n |\Delta_{ij}|\f]
       * \f[ \Delta_{min} = \min_{1\le i\le n, i+1\le j\le n} |\Delta_{ij}|\f]
       * \f[ \Delta_{max} = \max_{1\le i\le n, i+1\le j\le n}|\Delta_{ij}|\f]
       * and \f$ \Delta_{ij} \f$ is the cost function differnce between the step i and step j
       * of the random walk.
       */
      IT7(double k, double l1, double l1p, double l2, double l2p, size_t n, double step_size,
          NadirCostFunction &fnc, Eigen::VectorXd &par)
          : ITx(ITxCase::IT2), _k(k), _l1(l1), _l1p(l1p), _l2(l2), _l2p(l2p), _n(n),
            _step_size(step_size), _fnc(fnc), _par(par)
      {
         if (_l1 < 0 || _l1 > 1) throw std::invalid_argument("IT7: l1 must be in [0, 1]");
         if (_l1p < 0 || _l1p > 1) throw std::invalid_argument("IT7: l1p must be in [0, 1]");
         if (_l2 < 0 || _l2 > 1) throw std::invalid_argument("IT7: l2 must be in [0, 1]");
         if (_l2p < 0 || _l2p > 1) throw std::invalid_argument("IT7: l2p must be in [0, 1]");
         if (_l1 + _l1p > 1) throw std::invalid_argument("IT7: l1+l1p must be <=1");
         if (_l2 + _l2p > 1) throw std::invalid_argument("IT7: l2+l2p must be <=1");
      };

      /// Caller
      std::pair<double, double> operator()() override;

   private:
      /// \name Cached variables
      ///@{
      double _k;
      double _l1, _l1p, _l2, _l2p;
      size_t _n;
      double _step_size;
      NadirCostFunction &_fnc;
      Eigen::VectorXd _par;
      ///@}
};

} // namespace nadir
#endif // NADIR_ITX_SIMULATED_ANNEALING_H