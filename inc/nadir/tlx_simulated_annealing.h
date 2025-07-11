#ifndef NADIR_TLX_SIMULATED_ANNEALING_H
#define NADIR_TLX_SIMULATED_ANNEALING_H

#include "nadir/abstract_classes.h"
#include "nadir/context_simulated_annealing.h"

/**
 * @file tlx_simulated_annealing.h
 *
 * TLx components for the simulated
 * annealing, as presented in Computers and Operations Research 104 (2019) 191–206
 * (https://doi.org/10.1016/j.cor.2018.12.015)
 *
 * \note
 */

namespace nadir
{
enum class TLxCase : unsigned {
   TL1, /* Fix */
   TL2, /* Fix, = k * (6*width)^n_par */
   TL3, /* Fix, = k * (6*width)^(2n_par) */
   TL4, /* Fix  = k * n_par */
   TL5, /* Fix  = k * n_par^2 */
   // TL6, /* Not implemented, depends on number of accepted moves, which is not provided */
   // TL7, /* Not implemented, depends on number of accepted moves, which is not provided */
   TL8,  /* Adaptive = L + k */
   TL9,  /* Adaptive = L * k */
   TL10, /* Adaptive = k / L */
   TL11, /* Adaptive = L^(1/a) */
   CUSTOM,
};

class TLx
{
   public:
      const TLxCase tag;

      virtual ~TLx() = default;

      virtual size_t operator()(const SimAnnContext &) = 0;

   protected:
      TLx(TLxCase which) : tag(which)
      {
      }
};

class TL1 : public TLx
{
   public:
      TL1(size_t k) : TLx(TLxCase::TL1), _k(k) {};
      size_t operator()(const SimAnnContext &) override;

   private:
      size_t _k;
};

// Here width is assumed to be the width of the Gaussian for the NEx
class TL2 : public TLx
{
   public:
      TL2(size_t k, double width, size_t n_par) : TLx(TLxCase::TL2), _k(k)
      {
         _width = pow_n(6. * width, n_par); // Cube volume in n_par dim
      };
      size_t operator()(const SimAnnContext &) override;

   private:
      size_t _k, _width;
};

class TL3 : public TLx
{
   public:
      TL3(size_t k, double width, size_t n_par) : TLx(TLxCase::TL3), _k(k)
      {
         _width = pow_n(6. * width, 2 * n_par); // Cube volume squared in n_par dim
      };
      size_t operator()(const SimAnnContext &) override;

   private:
      size_t _k, _width;
};

class TL4 : public TLx
{
   public:
      TL4(size_t k, size_t n_par) : TLx(TLxCase::TL4), _k(k * n_par) {};
      size_t operator()(const SimAnnContext &) override;

   private:
      size_t _k;
};

class TL5 : public TLx
{
   public:
      TL5(size_t k, size_t n_par) : TLx(TLxCase::TL5), _k(k * n_par * n_par) {};
      size_t operator()(const SimAnnContext &) override;

   private:
      size_t _k;
};

class TL8 : public TLx
{
   public:
      TL8(size_t k) : TLx(TLxCase::TL8), _k(k) {};
      size_t operator()(const SimAnnContext &) override;

   private:
      size_t _k;
};

class TL9 : public TLx
{
   public:
      TL9(size_t k) : TLx(TLxCase::TL9), _k(k) {};
      size_t operator()(const SimAnnContext &) override;

   private:
      size_t _k;
};

class TL10 : public TLx
{
   public:
      TL10(size_t k) : TLx(TLxCase::TL10), _k(k) {};
      size_t operator()(const SimAnnContext &) override;

   private:
      size_t _k;
};

class TL11 : public TLx
{
   public:
      TL11(double a) : TLx(TLxCase::TL11), _a(a)
      {
         if (a <= 0 || a >= 1) throw std::invalid_argument("TL11 alpha must be in (0, 1)");
      };
      size_t operator()(const SimAnnContext &) override;

   private:
      double _a;
};

} // namespace nadir
#endif // NADIR_TLX_SIMULATED_ANNEALING_H