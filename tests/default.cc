#include <nadir/nadir.h>
#include <iostream>
int main()
{
   double x = 2;
   auto fn  = [&x](Eigen::VectorXd &g) -> void { g(0) = 2. * x * exp(-x * x); };
   nadir::Adam adam(fn, {&x});
   for (size_t i = 0; i < 500; i++) {
      adam.step();
   }
   std::cout << x << std::endl;
   return 0;
}