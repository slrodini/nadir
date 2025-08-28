#ifndef NADIR_RAN2_H
#define NADIR_RAN2_H

#include <cstddef>

/**
 * @file ran2.h
 * @brief Implementation of ran2 random number generation from Numerical Recipes
 */

namespace nadir
{

/// Set the seed. Can be called only once, before any other function is called
void set_seed(long int seed);
/// Uniform random number in [0, 1)
double _random_uniform();
/// Normal random number (mean = 0, std = 1)
double _random_normal();

/// Random int in [0, n)
size_t _random_uint(size_t n);

double _random_cauchy();
double _random_cauchy(double m, double s);

} // namespace nadir

#endif // NADIR_RAN2_H