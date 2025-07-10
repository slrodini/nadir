#include <nadir/nadir.h>
#include <iostream>
#include <algorithm>
#include <random>

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

using adam_var = nadir::Adam::ADAM_VARIANT;

std::pair<double, double> test_gaussian();
std::pair<double, double> test_rosenbrock();
std::pair<double, double> test_schwefel();

int main()
{

   std::vector<std::function<std::pair<double, double>()>> tests;
   tests.push_back(test_gaussian);
   tests.push_back(test_rosenbrock);
   tests.push_back(test_schwefel);

   for (auto &test : tests) {
      auto res = test();
      std::cout << res.first << " " << res.second << std::endl;
   }
}

std::pair<double, double> rosenbrock(double x0, double y0);
std::pair<double, double> test_rosenbrock()
{
   std::vector<double> deltas_f;
   std::vector<double> deltas_p;
   for (double i = -1000; i < 1000; i += 100) {
      for (double j = -1000; j < 1000; j += 100) {
         auto [delta_f, delta_p] = rosenbrock(2., 2.);
         // std::cout << delta_f << " " << delta_p << std::endl;
         deltas_f.push_back(std::fabs(delta_f));
         deltas_p.push_back(delta_p);
      }
   }
   auto it_f = std::max_element(deltas_f.begin(), deltas_f.end());
   auto it_p = std::max_element(deltas_f.begin(), deltas_f.end());

   return {*it_f, *it_p};
}

std::pair<double, double> schwefel(long int d, std::ranlux48 &engine);
std::pair<double, double> test_schwefel()
{
   std::vector<double> deltas_f;
   std::vector<double> deltas_p;
   std::ranlux48 engine(314);

   for (long int d = 2; d < 3; d++) {
      for (size_t i = 0; i < 1; i++) {
         auto [delta_f, delta_p] = schwefel(d, engine);
         deltas_f.push_back(std::fabs(delta_f));
         deltas_p.push_back(delta_p);
      }
   }
   auto it_f = std::max_element(deltas_f.begin(), deltas_f.end());
   auto it_p = std::max_element(deltas_f.begin(), deltas_f.end());
   return {*it_f, *it_p};
}

std::pair<double, double> test_gaussian()
{
   Eigen::VectorXd x(1);
   x(0) = 2;

   auto fn = [](const Eigen::VectorXd &x, double &v, Eigen::VectorXd &g) -> void {
      v    = exp(-x(0) * x(0));
      g(0) = 2. * x(0) * exp(-x(0) * x(0));
   };

   TestCostFunction fn_cost(fn);
   nadir::Adam adam(fn_cost, x);
   nadir::Adam::MetaParameters mp = nadir::Adam::MetaParameters{
       .max_it          = 1000,
       .alpha           = 0.1,
       .diff_value_toll = 1.0e-16,
   };

   adam.SetMetaParameters(mp);
   nadir::Adam::STATUS st = adam.minimize();
   (void)st;
   x = adam.GetParameters();
   return {exp(-x(0) * x(0)) - 1, std::fabs(x(0))};
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
   nadir::Adam adam(fn_cost, p);
   nadir::Adam::MetaParameters mp = nadir::Adam::MetaParameters{
       .variant         = adam_var::RADAM,
       .max_it          = 3000,
       .alpha           = 1.,
       .eps             = 1.0e-4,
       .grad_toll       = 1.0e-16,
       .diff_value_toll = 1.0e-16,
       .lambda          = 0.0001,
   };
   auto cosAnnSched = [&mp](size_t t) {
      double tp        = static_cast<double>(t);
      double Tp        = static_cast<double>(mp.max_it);
      double alpha_max = 1.;
      double alpha_min = 1.0e-4;
      return alpha_min + 0.5 * (alpha_max - alpha_min) * (1. + cos(M_PI * tp / Tp));
   };
   adam.SetScheduler(cosAnnSched);

   adam.SetMetaParameters(mp);
   nadir::Adam::STATUS st = adam.minimize();
   (void)st;
   Eigen::VectorXd diff             = (p_true - adam.GetParameters());
   std::pair<double, double> result = {fn(adam.GetParameters()), sqrt(diff.squaredNorm())};

   return result;
}

std::pair<double, double> schwefel(long int d, std::ranlux48 &engine)
{
   Eigen::VectorXd p(d);
   std::uniform_real_distribution<double> dist(-500., 500.);
   for (long int i = 0; i < d; i++) {
      p(i) = dist(engine);
      std::cout << p(i) << std::endl;
   }

   Eigen::VectorXd p_true(d);
   for (long int i = 0; i < d; i++) {
      p_true(i) = 420.9687;
   }

   auto fn = [&d](const Eigen::VectorXd &p) -> double {
      double tmp = 418.9829 * static_cast<double>(d);
      for (long int i = 0; i < d; i++) {
         tmp -= p(i) * sin(sqrt(fabs(p(i))));
      }
      return tmp;
   };
   auto sign = [](double x) -> double {
      return x > 0. ? 1. : (x < 0. ? -1. : 0.);
   };

   auto fn_grad = [&sign, &d](const Eigen::VectorXd &p, double &v, Eigen::VectorXd &g) -> void {
      v = 418.9829 * static_cast<double>(d);
      for (long int i = 0; i < d; i++) {
         double sqrt_xi = sqrt(fabs(p(i)));
         v -= p(i) * sin(sqrt_xi);
         g(i) = -sign(p(i)) * 0.5 * sqrt_xi * cos(sqrt_xi) - sin(sqrt_xi);
      }
   };

   TestCostFunction fn_cost(fn_grad);
   nadir::Adam adam(fn_cost, p);
   nadir::Adam::MetaParameters mp = nadir::Adam::MetaParameters{
       .variant         = adam_var::RADAM,
       .max_it          = 3000,
       .alpha           = 1.,
       .eps             = 1.0e-4,
       .grad_toll       = 1.0e-16,
       .diff_value_toll = 1.0e-16,
       .lambda          = 0.0001,
   };
   size_t T         = 100;
   double Tp        = static_cast<double>(T);
   auto cosAnnSched = [&T, &Tp](size_t t) {
      double tp        = static_cast<double>(t);
      double alpha_max = 1.;
      double alpha_min = 1.0e-5;

      if (t % T == 0) {
         T *= 2;
         Tp *= 2;
      }
      tp = static_cast<double>(t % T);
      return alpha_min + 0.5 * (alpha_max - alpha_min) * (1. + cos(M_PI * tp / Tp));
   };
   adam.SetScheduler(cosAnnSched);

   adam.SetMetaParameters(mp);
   nadir::Adam::STATUS st = adam.minimize();
   (void)st;
   Eigen::VectorXd diff = (p_true - adam.GetParameters());

   // adam.FlusToStdout();
   std::cout << "---\n" << adam.GetParameters() << "\n---\n";
   return {fn(adam.GetParameters()), sqrt(diff.squaredNorm())};
}
