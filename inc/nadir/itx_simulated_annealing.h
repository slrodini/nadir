#ifndef NADIR_ITX_SIMULATED_ANNEALING_H
#define NADIR_ITX_SIMULATED_ANNEALING_H

#include "nadir/abstract_classes.h"

/**
 * @file itx_simulated_annealing.h
 *
 * ITx components for the simulated
 * annealing, as presented in Computers and Operations Research 104 (2019) 191–206
 * (https://doi.org/10.1016/j.cor.2018.12.015)
 *
 */

namespace nadir
{

enum class ITxCase {
   IT1, /* fixed initial temperature T0 = k*/
   IT2, /* T0 proportional to initial cost T0 = k C(0) */
   IT3, /* T0 proportional to max gap in consecutive costs in a random walk */
   IT4, /* T0 proportional to average gap in consecutive costs in a random walk */
   IT6, /* T0 as fixed probability p0 in a random walk */
   IT7, /* T0 from random walk, using min, max and average of gaps */
   CUSTOM,
};

class ITx
{
   public:
      const ITxCase tag;

      virtual ~ITx() = default;

      // Returns [T0, Tf]
      virtual std::pair<double, double> operator()() = 0;

   protected:
      ITx(ITxCase which) : tag(which)
      {
      }
};

/// Implement constant T0
class IT1 : public ITx
{
   public:
      IT1(double k, double Tf) : ITx(ITxCase::IT1), _k(k), _Tf(Tf) {};
      std::pair<double, double> operator()() override;

   private:
      double _k, _Tf;
};

///
class IT2 : public ITx
{
   public:
      IT2(double k, double Tf, NadirCostFunction &fnc, Eigen::VectorXd &par)
          : ITx(ITxCase::IT2), _k(k), _Tf(Tf), _fnc(fnc), _par(par) {};

      std::pair<double, double> operator()() override;

   private:
      double _k, _Tf;
      NadirCostFunction &_fnc;
      Eigen::VectorXd &_par;
};

void itx_random_walk(NadirCostFunction &fnc, Eigen::VectorXd &par, size_t n, double step_size,
                     std::vector<double> &out);

/// Min-Max random walk
class IT3 : public ITx
{
   public:
      IT3(double k, size_t n, double step_size, NadirCostFunction &fnc, Eigen::VectorXd &par)
          : ITx(ITxCase::IT2), _k(k), _n(n), _step_size(step_size), _fnc(fnc), _par(par) {};

      std::pair<double, double> operator()() override;

   private:
      double _k;
      size_t _n;
      double _step_size;
      NadirCostFunction &_fnc;
      Eigen::VectorXd _par;
};

/// Average random walk
class IT4 : public ITx
{
   public:
      IT4(double k, double Tf, size_t n, double step_size, NadirCostFunction &fnc,
          Eigen::VectorXd &par)
          : ITx(ITxCase::IT2), _k(k), _Tf(Tf), _n(n), _step_size(step_size), _fnc(fnc),
            _par(par) {};

      std::pair<double, double> operator()() override;

   private:
      double _k, _Tf;
      size_t _n;
      double _step_size;
      NadirCostFunction &_fnc;
      Eigen::VectorXd _par;
};

/// Average probability random walk
class IT6 : public ITx
{
   public:
      IT6(double k, double Tf, double p0, size_t n, double step_size, NadirCostFunction &fnc,
          Eigen::VectorXd &par)
          : ITx(ITxCase::IT2), _k(k), _Tf(Tf), _p0(p0), _n(n), _step_size(step_size), _fnc(fnc),
            _par(par)
      {
         if (_p0 <= 0 || _p0 >= 1) throw std::invalid_argument("IT6: p0 must be in (0, 1)");
      };

      std::pair<double, double> operator()() override;

   private:
      double _k, _Tf;
      double _p0;
      size_t _n;
      double _step_size;
      NadirCostFunction &_fnc;
      Eigen::VectorXd _par;
};

/// Min-Max-Average random walk
class IT7 : public ITx
{
   public:
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

      std::pair<double, double> operator()() override;

   private:
      double _k;
      double _l1, _l1p, _l2, _l2p;
      size_t _n;
      double _step_size;
      NadirCostFunction &_fnc;
      Eigen::VectorXd _par;
};

} // namespace nadir
#endif // NADIR_ITX_SIMULATED_ANNEALING_H