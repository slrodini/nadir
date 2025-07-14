#include <nadir/nadir.h>
#include <iostream>
#include <algorithm>

class TestCostFunction : public nadir::NadirCostFunction
{
   public:
      TestCostFunction(
          std::function<void(const Eigen::VectorXd &, double &, Eigen::VectorXd &)> fnc)
          : _fnc(fnc)
      {
      }
      void Evaluate(const Eigen::VectorXd &p, double &v, Eigen::VectorXd &g) override
      {
         _fnc(p, v, g);
      }

   private:
      std::function<void(const Eigen::VectorXd &, double &, Eigen::VectorXd &)> _fnc;
};

std::pair<double, double> rosenbrock(double x0, double y0);
std::pair<double, double> test_rosenbrock();

int main()
{

   // std::vector<std::function<std::pair<double, double>()>> tests;
   // tests.push_back(test_rosenbrock);

   // for (auto &test : tests) {
   //    auto res = test();
   //    std::cout << res.first << " " << res.second << std::endl;
   // }

   std::pair<double, double> res = rosenbrock(2, 2);
   std::cout << res.first << " " << res.second << std::endl;

   return 0;
}

std::pair<double, double> test_rosenbrock()
{
   std::vector<double> deltas_f;
   std::vector<double> deltas_p;
   for (double i = -10; i < 10; i += 1) {
      for (double j = -10; j < 10; j += 1) {
         auto [delta_f, delta_p] = rosenbrock(i, j);
         // std::cout << delta_f << " " << delta_p << std::endl;
         deltas_f.push_back(std::fabs(delta_f));
         deltas_p.push_back(delta_p);
      }
   }
   auto it_f = std::max_element(deltas_f.begin(), deltas_f.end());
   auto it_p = std::max_element(deltas_f.begin(), deltas_f.end());

   return {*it_f, *it_p};
}

std::pair<double, double> rosenbrock(double x0, double y0)
{
   Eigen::VectorXd p(2);
   p(0) = x0;
   p(1) = y0;

   auto sq = [](double x) {
      return x * x;
   };

   const double a = 1.;
   const double b = 100.;
   Eigen::VectorXd p_true(2);
   p_true << a, a * a;

   auto fn = [&sq, &a, &b](const Eigen::VectorXd &p) -> double {
      return sq(p(0) - a) + b * sq(p(1) - sq(p(0)));
   };

   auto fn_grad = [&sq, &a, &b](const Eigen::VectorXd &p, double &v, Eigen::VectorXd &g) -> void {
      const double xma  = p(0) - a;
      const double ymx2 = p(1) - sq(p(0));

      v    = sq(xma) + b * sq(ymx2);
      g(1) = 2. * b * ymx2;
      g(0) = 2. * xma - 4. * b * p(0) * ymx2;
   };

   TestCostFunction fn_cost(fn_grad);
   nadir::Shampoo::MetaParameters mp = nadir::Shampoo::MetaParameters{
       .max_it          = 2000,
       .beta            = 0.5,
       .lambda          = 0.01,
       .eps             = 1.0e-8,
       .grad_toll       = 1.0e-16,
       .diff_value_toll = 1.0e-16,
   };
   nadir::Shampoo shampoo(mp, fn_cost, p);

   nadir::Minimizer::STATUS st = shampoo.minimize();

   if (st == nadir::Minimizer::STATUS::ABORT) {
      throw std::runtime_error("Failed minimization with shampoo!");
   }
   Eigen::VectorXd diff             = (p_true - shampoo.GetParameters());
   std::pair<double, double> result = {fn(shampoo.GetParameters()), sqrt(diff.squaredNorm())};
   shampoo.FlusToStdout();
   return result;
}
