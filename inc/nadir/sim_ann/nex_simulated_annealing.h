#ifndef NADIR_NEX_SIMULATED_ANNEALING_H
#define NADIR_NEX_SIMULATED_ANNEALING_H

#include "nadir/abstract_classes.h"
#include "nadir/sim_ann/context_simulated_annealing.h"

/**
 * @file nex_simulated_annealing.h
 *
 * NEx components for the simulated
 * annealing, as presented in Computers and Operations Research 104 (2019) 191–206
 * (https://doi.org/10.1016/j.cor.2018.12.015)
 *
 * \note
 */

namespace nadir
{
enum class NExCase {
   // NE1, /* Random neighbour; this seems to be NE3 with just one step*/
   // NE2, /* Sequential neighbour, not sure what it should be */
   NE3, /* Best of k neighbours */
   NE4, /* First improvement in k neighbours */
   NE5, /* Same as NE3, but with mono-parameter update */
   NE6, /* Same as NE4, but with mono-parameter update */

};

class NEx
{
   public:
      const NExCase tag;

      virtual ~NEx() = default;
      // Note: this call will return the cost of the neighbour and store the
      // corresponding parameters into the _pars vector.
      virtual void operator()(SimAnnContext &context, NadirCostFunction &fnc,
                              const Eigen::VectorXd &_incumbent_pars, Eigen::VectorXd &_pars) = 0;

   protected:
      NEx(NExCase which) : tag(which)
      {
      }
};

class NE3 : public NEx
{
   public:
      NE3(double width, size_t n = 1) : NEx(NExCase::NE3), _width(width), _n(n) {};
      void operator()(SimAnnContext &context, NadirCostFunction &fnc,
                      const Eigen::VectorXd &_incumbent_pars, Eigen::VectorXd &_pars) override;

   private:
      double _width;
      size_t _n;
      Eigen::VectorXd _best;
};

class NE4 : public NEx
{
   public:
      NE4(double width, size_t n) : NEx(NExCase::NE4), _width(width), _n(n) {};
      void operator()(SimAnnContext &context, NadirCostFunction &fnc,
                      const Eigen::VectorXd &_incumbent_pars, Eigen::VectorXd &_pars) override;

   private:
      double _width;
      size_t _n;
      Eigen::VectorXd _best;
};

class NE5 : public NEx
{
   public:
      NE5(double width, size_t n = 1) : NEx(NExCase::NE5), _width(width), _n(n) {};
      void operator()(SimAnnContext &context, NadirCostFunction &fnc,
                      const Eigen::VectorXd &_incumbent_pars, Eigen::VectorXd &_pars) override;

   private:
      double _width;
      size_t _n;
      Eigen::VectorXd _best;
};

class NE6 : public NEx
{
   public:
      NE6(double width, size_t n) : NEx(NExCase::NE6), _width(width), _n(n) {};
      void operator()(SimAnnContext &context, NadirCostFunction &fnc,
                      const Eigen::VectorXd &_incumbent_pars, Eigen::VectorXd &_pars) override;

   private:
      double _width;
      size_t _n;
      Eigen::VectorXd _best;
};

} // namespace nadir
#endif // NADIR_NEX_SIMULATED_ANNEALING_H