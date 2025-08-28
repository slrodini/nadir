#include <nadir/nadir.h>

class TestCostFunction : public nadir::NadirCostFunction
{
   public:
      TestCostFunction(std::function<void(const Eigen::VectorXd &, double &)> fnc) : _fnc(fnc)
      {
      }
      void Evaluate(const Eigen::VectorXd &p, double &v) override
      {
         _fnc(p, v);
      }

   private:
      std::function<void(const Eigen::VectorXd &, double &)> _fnc;
};

std::pair<double, double> rosenbrock(double x0, double y0);
std::pair<double, double> test_rosenbrock();

int main()
{

   std::vector<std::function<std::pair<double, double>()>> tests;
   tests.push_back(test_rosenbrock);

   for (auto &test : tests) {
      auto res = test();
      std::cout << res.first << " " << res.second << std::endl;
   }

   return 0;
}

std::pair<double, double> test_rosenbrock()
{
   std::vector<double> deltas_f;
   std::vector<double> deltas_p;
   for (double i = -10; i < 10; i += 1) {
      for (double j = -10; j < 10; j += 1) {
         auto [delta_f, delta_p] = rosenbrock(i, j);
         std::cout << i << " " << j << " " << delta_f << " " << delta_p << std::endl;
         deltas_f.push_back(std::fabs(delta_f));
         deltas_p.push_back(delta_p);
      }
   }
   auto it_f = std::max_element(deltas_f.begin(), deltas_f.end());
   auto it_p = std::max_element(deltas_p.begin(), deltas_p.end());

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

   auto fn = [&sq, &a, &b](const Eigen::VectorXd &p, double &r) -> void {
      r = sq(p(0) - a) + b * sq(p(1) - sq(p(0)));
   };

   TestCostFunction fn_cost(fn);

   nadir::CMA_ES diffevo({.sigma = 1., .max_fnc_eval = 100000, .max_iter = 200}, fn_cost, p);

   // nadir::Minimizer::STATUS st = diffevo.minimize();
   auto st = diffevo.ipop_minimize({.sigma_ref = 10., .max_iter = 20});
   (void)st;
   Eigen::VectorXd diff = (p_true - diffevo.GetParameters());

   double tmp = 0;
   fn(diffevo.GetParameters(), tmp);

   std::pair<double, double> result = {tmp, sqrt(diff.squaredNorm())};

   return result;
}