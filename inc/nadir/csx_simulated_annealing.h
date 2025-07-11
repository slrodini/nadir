#ifndef NADIR_CSX_SIMULATED_ANNEALING_H
#define NADIR_CSX_SIMULATED_ANNEALING_H

#include "nadir/context_simulated_annealing.h"

/**
 * @file csx_simulated_annealing.h
 *
 * CSx components for the simulated
 * annealing, as presented in Computers and Operations Research 104 (2019) 191–206
 * (https://doi.org/10.1016/j.cor.2018.12.015)
 *
 * \note
 */

namespace nadir
{
enum class CSxCase : unsigned {
   CS1, /* alpha T^beta */
   CS2, /* alpha T*/
   CS3, /* alpha / log(b+i)*/
   CS4, /* alpha / (b + log(i))*/
   CS5, /* T / (a + b T)*/
   // CS6, /* Not implemented, must have memory of discarded steps */
   CS7, /* a / (1 + b T) */
   // CS8, /* Not implemented, has dependence over T0, Tf, and max_iter */
   CS9,  /* max(T - _a, 0.) */
   CS10, /* T=const WARNING: use it only with max_iter termination schemes! */
   CS11, /* Random in [Tx, x Tx] WARNING: use it only with max_iter termination schemes! */
   // CS12, /* Not implemented, depends on current accepted outcome */
   // CS13, /* Not implemented, depends on current accepted outcome */
   CUSTOM,

};

class CSx
{
   public:
      const CSxCase tag;

      virtual ~CSx() = default;

      virtual void operator()(SimAnnContext &) = 0;

   protected:
      CSx(CSxCase which) : tag(which)
      {
      }
};

class CS1 : public CSx
{
   public:
      CS1(double alpha, double beta) : CSx(CSxCase::CS1), _alpha(alpha), _beta(beta)
      {
         if (alpha <= 0 || alpha >= 1) throw std::invalid_argument("CS1 alpha must be in (0, 1)");
         if (beta <= 0 || beta >= 1) throw std::invalid_argument("CS1 beta must be in (0, 1)");
      };
      void operator()(SimAnnContext &) override;

   private:
      double _alpha, _beta;
      /*
       Note: in the paper they quote alpha beta^T, but this make little sense because
       this law would converge to a finite value (non-zero) and, for some typical choice of
       alpha and beta it does so very rapidly, like in ~5 steps.
       Therefore I implemented the much more same alpha T^beta version.
      */
};

class CS2 : public CSx
{
   public:
      CS2(double alpha) : CSx(CSxCase::CS2), _alpha(alpha)
      {
         if (alpha <= 0 || alpha >= 1) throw std::invalid_argument("CS2 alpha must be in (0, 1)");
      };
      void operator()(SimAnnContext &) override;

   private:
      double _alpha;
};

class CS3 : public CSx
{
   public:
      CS3(double alpha, double beta) : CSx(CSxCase::CS3), _alpha(alpha), _beta(beta)
      {
         if (alpha <= 0) throw std::invalid_argument("CS3 alpha must be >0");
         if (beta <= 0) throw std::invalid_argument("CS3 beta must be >0");
      };
      void operator()(SimAnnContext &) override;

   private:
      double _alpha, _beta;
};

class CS4 : public CSx
{
   public:
      CS4(double alpha, double beta) : CSx(CSxCase::CS4), _alpha(alpha), _beta(beta)
      {
         if (alpha <= 0) throw std::invalid_argument("CS3 alpha must be >0");
         if (beta <= 0) throw std::invalid_argument("CS3 beta must be >0");
      };
      void operator()(SimAnnContext &) override;

   private:
      double _alpha, _beta;
};

class CS5 : public CSx
{
   public:
      CS5(double alpha, double beta) : CSx(CSxCase::CS5), _alpha(alpha), _beta(beta) {};
      void operator()(SimAnnContext &) override;

   private:
      double _alpha, _beta;
};

class CS7 : public CSx
{
   public:
      CS7(double alpha, double beta) : CSx(CSxCase::CS7), _alpha(alpha), _beta(beta)
      {
         if (alpha <= 0) throw std::invalid_argument("CS7 alpha must be >0");
      };
      void operator()(SimAnnContext &) override;

   private:
      double _alpha, _beta;
};

class CS9 : public CSx
{
   public:
      CS9(double a) : CSx(CSxCase::CS9), _a(a) {};
      void operator()(SimAnnContext &) override;

   private:
      double _a;
};

class CS10 : public CSx
{
   public:
      CS10(double Tx) : CSx(CSxCase::CS10), _Tx(Tx) {};
      void operator()(SimAnnContext &) override;

   private:
      double _Tx;
};

class CS11 : public CSx
{
   public:
      CS11(double Tx, double a) : CSx(CSxCase::CS11), _Tx(Tx), _a(a)
      {
         if (a <= 1) throw std::invalid_argument("CS11 alpha must be >1");
      };
      void operator()(SimAnnContext &) override;

   private:
      double _Tx, _a;
};

} // namespace nadir
#endif