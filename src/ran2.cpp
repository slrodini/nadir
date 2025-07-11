#include <mutex>
#include <cmath>

// This is a file included in simulated_annealing.cc
// IT IS NOT TO BE COMPILED AS A SEPARATE TRANSLATION UNIT, NOR IT IS TO BE USED ANYWHERE ELSE
namespace
{
#define IM1 2147483563
#define IM2 2147483399
#define AM (1.0 / IM1)
#define IMM1 (IM1 - 1)
#define IA1 40014
#define IA2 40692
#define IQ1 53668
#define IQ2 52774
#define IR1 12211
#define IR2 3791
#define NTAB 32
#define NDIV (1 + IMM1 / NTAB)
#define EPS 1.2e-7
#define RNMX (1.0 - EPS)

static int64_t _idum = -2;
static int64_t idum2 = 123456789L;
static int64_t iy    = 0;
static int64_t iv[NTAB];
static bool seeded = false;

std::mutex _rand_mtx, _rand_mtx_n;

// Implementation of ran2 from Numerical Recipies
double _random_uniform()
{
   double res;
   {
      std::unique_lock<std::mutex> lock(_rand_mtx);
      if (!seeded) {
         _idum  = 2;
         seeded = true;
      }
      int j;
      int64_t k;
      double temp;

      if (_idum <= 0) {
         if (-(_idum) < 1) _idum = 1;
         else _idum = -(_idum);
         idum2 = (_idum);
         for (j = NTAB + 7; j >= 0; j--) {
            k     = (_idum) / IQ1;
            _idum = IA1 * (_idum - k * IQ1) - k * IR1;
            if (_idum < 0) _idum += IM1;
            if (j < NTAB) iv[j] = _idum;
         }
         iy = iv[0];
      }
      k     = (_idum) / IQ1;
      _idum = IA1 * (_idum - k * IQ1) - k * IR1;
      if (_idum < 0) _idum += IM1;
      k     = idum2 / IQ2;
      idum2 = IA2 * (idum2 - k * IQ2) - k * IR2;
      if (idum2 < 0) idum2 += IM2;
      j     = iy / NDIV;
      iy    = iv[j] - idum2;
      iv[j] = _idum;
      if (iy < 1) iy += IMM1;
      if ((temp = AM * iy) > RNMX) res = RNMX;
      else res = temp;
   }
   return res;
}

#undef IM1
#undef IM2
#undef AM
#undef IMM1
#undef IA1
#undef IA2
#undef IQ1
#undef IQ2
#undef IR1
#undef IR2
#undef NTAB
#undef NDIV
#undef EPS
#undef RNMX

double _random_normal()
{
   static bool is_set = false;
   static double gset = 0.0;
   double deviate;
   {
      std::unique_lock<std::mutex> lock(_rand_mtx_n);
      double fac, rsq, v1, v2;
      if (!is_set) {
         do {
            v1  = 2.0 * _random_uniform() - 1.0;
            v2  = 2.0 * _random_uniform() - 1.0;
            rsq = v1 * v1 + v2 * v2;
         } while (rsq >= 1.0 || rsq == 0.0);
         fac = sqrt(-2.0 * log(rsq) / rsq);

         gset    = v1 * fac;
         deviate = v2 * fac;
         is_set  = true;
      } else {
         is_set  = false;
         deviate = gset;
      }
   }

   return deviate;
}
} // namespace