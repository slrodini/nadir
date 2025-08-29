#ifndef NADIR_SCX_SIMULATED_ANNEALING_H
#define NADIR_SCX_SIMULATED_ANNEALING_H

#include "nadir/sim_ann/context_simulated_annealing.h"

/**
 * @file scx_simulated_annealing.h
 *
 * SCx components for the simulated
 * annealing, as presented in Computers and Operations Research 104 (2019) 191–206
 * (https://doi.org/10.1016/j.cor.2018.12.015)
 *
 * \note Not all strategies are implemented, because not all of them scale well
 * with the problem computational complexity in a general case.
 * For instance, SC1, which is to stop after a given amount of time, is not
 * well suited for a general minimization, because it depdends strongly on which
 * machine the minimization is run.
 */

namespace nadir
{

enum class SCxCase {
   SC2, /* Fix total candidate moves*/
   SC3, /* min temperature */
   SC4, /* max cooling steps */
   // SC5, /* max temperature restarts, not provided */
   SC6, /* max failed consecutive candidate moves */
   SC7, /* min total acceptance rate */

};

class SCx
{
   public:
      const SCxCase tag;

      virtual ~SCx() = default;

      virtual bool operator()(const SimAnnContext &) = 0;

   protected:
      SCx(SCxCase which) : tag(which)
      {
      }
};

///
class SC2 : public SCx
{
   public:
      SC2(size_t n) : SCx(SCxCase::SC2), _n(n) {};
      bool operator()(const SimAnnContext &) override;

   private:
      size_t _n;
};

class SC3 : public SCx
{
   public:
      SC3() : SCx(SCxCase::SC3) {};
      bool operator()(const SimAnnContext &) override;
};

class SC4 : public SCx
{
   public:
      SC4(size_t n) : SCx(SCxCase::SC4), _n(n) {};
      bool operator()(const SimAnnContext &) override;

   private:
      size_t _n;
};

class SC6 : public SCx
{
   public:
      SC6(size_t n) : SCx(SCxCase::SC6), _n(n) {};
      bool operator()(const SimAnnContext &) override;

   private:
      size_t _n;
};

class SC7 : public SCx
{
   public:
      SC7(double ar) : SCx(SCxCase::SC6), _ar(ar) {};
      bool operator()(const SimAnnContext &) override;

   private:
      double _ar;
};

} // namespace nadir
#endif // NADIR_SCX_SIMULATED_ANNEALING_H