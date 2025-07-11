#ifndef NADIR_ACX_SIMULATED_ANNEALING_H
#define NADIR_ACX_SIMULATED_ANNEALING_H

#include "nadir/context_simulated_annealing.h"

/**
 * @file acx_simulated_annealing.h
 *
 * ACx components for the simulated
 * annealing, as presented in Computers and Operations Research 104 (2019) 191–206
 * (https://doi.org/10.1016/j.cor.2018.12.015)
 *
 * \note
 */

namespace nadir
{
enum class ACxCase : unsigned {
   AC1, /* Metropolis */
   AC2, /* Cutted Metropolis if exp(-Delta/T) is too small */
   AC3, /* Metropolis with rejection based on quality control function */
   AC4, /* Modified Metropolis alghorithm */
   AC5, /* Geometric (power-law) */
   AC6, /* Deterministic temperature dependent cut-off */
   AC7, /* Non-comparative method, accept base on comparison of current with given phi(T, step) */
   // AC8, /* Uses best solution, not implemented */
   // AC9, /* Uses memory update, not yet provided by implementation */
   AC10, /* Accept only improving solutions */
   CUSTOM,

};

class ACx
{
   public:
      const ACxCase tag;

      virtual ~ACx() = default;
      // Note: parameter update based on the result is handled in the annealing class
      virtual bool operator()(const SimAnnContext &) = 0;

   protected:
      ACx(ACxCase which) : tag(which)
      {
      }
};

class AC1 : public ACx
{
   public:
      AC1() : ACx(ACxCase::AC1) {};
      bool operator()(const SimAnnContext &) override;
};

class AC2 : public ACx
{
   public:
      AC2(double eps) : ACx(ACxCase::AC2), _log_eps(log(eps)) {};
      bool operator()(const SimAnnContext &) override;

   private:
      double _log_eps;
};

class AC3 : public ACx
{
   public:
      AC3(std::function<double(double)> &q, double phi)
          : ACx(ACxCase::AC3), _quality(q), _phi(phi) {};
      bool operator()(const SimAnnContext &) override;

   private:
      std::function<double(double)> _quality;
      double _phi;
};

class AC4 : public ACx
{
   public:
      AC4(std::function<double(double)> &q, std::function<double(double)> &beta, double g)
          : ACx(ACxCase::AC4), _quality(q), _beta(beta), _g(g) {};
      bool operator()(const SimAnnContext &) override;

   private:
      std::function<double(double)> _quality;
      std::function<double(double)> _beta;
      double _g;
};

class AC5 : public ACx
{
   public:
      AC5(double p0, double r) : ACx(ACxCase::AC5), _p0(p0), _r(r), _k(1)
      {
         if (p0 <= 0 || p0 > 1) throw std::invalid_argument("AC5: p0 must be in (0, 1].");
         if (r <= 0 || r > 1) throw std::invalid_argument("AC5: r must be in (0, 1].");
      };
      bool operator()(const SimAnnContext &) override;

   private:
      double _p0, _r;
      size_t _k;
};

class AC6 : public ACx
{
   public:
      AC6(std::function<double(double)> &phi) : ACx(ACxCase::AC6), _phi(phi) {};
      bool operator()(const SimAnnContext &) override;

   private:
      std::function<double(double)> _phi;
};

class AC7 : public ACx
{
   public:
      AC7(std::function<double(double, size_t)> &phi) : ACx(ACxCase::AC7), _phi(phi) {};
      bool operator()(const SimAnnContext &) override;

   private:
      std::function<double(double, size_t)> _phi;
      size_t _k;
};

class AC10 : public ACx
{
   public:
      AC10() : ACx(ACxCase::AC10) {};
      bool operator()(const SimAnnContext &) override;
};

} // namespace nadir
#endif // NADIR_ACX_SIMULATED_ANNEALING_H