#pragma once

#include "nadir/nadir.h"

static Ran2 global_ran2;

struct Problem {
      std::string name = "_empty_";
      size_t D         = 0;
      double L = -3, U = 3;
      Eigen::VectorXd p_true;
      std::function<double(const Eigen::VectorXd &)> fnc;
      std::function<double(const Eigen::VectorXd &, Eigen::VectorXd &)> fnc_der;
};

struct ProblemWrapper : public nadir::NadirCostFunction {
      const Problem &problem;
      size_t evals = 0;

      ProblemWrapper(const Problem &p) : problem(p)
      {
      }

      void Evaluate(const Eigen::VectorXd &parameters, double &cost) override
      {
         evals++;
         cost = problem.fnc(parameters);
      }

      void Evaluate(const Eigen::VectorXd &parameters, double &cost,
                    Eigen::VectorXd &cost_derivatives) override
      {
         evals++;
         cost = problem.fnc_der(parameters, cost_derivatives);
      }
};

struct RunResult {
      std::string name = "_empty_";
      size_t D         = 0;
      size_t evals;
      Eigen::VectorXd p_true;
      Eigen::VectorXd p_best;
      double distance;
};

inline Problem make_rosenbrock(size_t d, double L = -3, double U = 3)
{
   if (L > U) std::swap(L, U);
   if (d <= 1) throw std::invalid_argument("Rosenbrock only defined for d>1");

   Problem P;
   P.name = "Rosenbrock-" + std::to_string(d);
   P.D    = d;
   P.L    = -10. + L;
   P.U    = 10. + U;

   Eigen::VectorXd p_true =
       Eigen::VectorXd::Constant(d, L) +
       (Eigen::VectorXd::Constant(d, U) - Eigen::VectorXd::Constant(d, L)).unaryExpr([](double) {
          return global_ran2.uniform();
       });
   P.p_true = p_true;

   P.fnc = [p_true, d](const Eigen::VectorXd &x) -> double {
      Eigen::VectorXd y = 1.0 + (x - p_true).array();
      double s          = 0;
      for (size_t i = 0; i < d - 1; i++) {
         s += (1.0 - y(i)) * (1.0 - y(i));
         s += 100. * (y(i + 1) - y(i) * y(i)) * (y(i + 1) - y(i) * y(i));
      }
      return s;
   };

   P.fnc_der = [p_true, d](const Eigen::VectorXd &x, Eigen::VectorXd &der) -> double {
      Eigen::VectorXd y = 1.0 + (x - p_true).array();
      double s          = 0;
      for (size_t i = 0; i < d - 1; i++) {
         s += (1.0 - y(i)) * (1.0 - y(i));
         s += 100. * (y(i + 1) - y(i) * y(i)) * (y(i + 1) - y(i) * y(i));
         der(i) = -2 * (1 - y(i)) - 400 * y(i) * (y(i + 1) - y(i) * y(i));
      }
      der(d - 1) = 200 * (y(d - 1) - y(d - 2) * y(d - 2));

      return s;
   };

   return P;
}

inline std::vector<Problem> make_suite(size_t D)
{
   std::vector<Problem> P;
   P.emplace_back(make_rosenbrock(D));
   return P;
}

inline RunResult run_one(nadir::Minimizer &minimizer, const Problem &P)
{
   RunResult res = RunResult{
       .name     = P.name,
       .D        = P.D,
       .evals    = 0,
       .p_true   = P.p_true,
       .p_best   = Eigen::VectorXd::Zero(P.p_true.size()),
       .distance = std::numeric_limits<double>::infinity(),
   };

   ProblemWrapper pw(P);
   Eigen::VectorXd start(P.p_true.size());

   for (long int i = 0; i < start.size(); i++) {
      start(i) = global_ran2.uniform(P.L, P.U);
   }

   minimizer.Reset(pw, start);
   std::cout << nadir::Minimizer::print_status(minimizer.minimize()) << std::endl;

   res.p_best = minimizer.GetParameters();
   res.evals  = pw.evals;

   res.distance = (res.p_best - res.p_true).norm() / sqrt((double)P.D);

   return res;
}

struct SuiteSummary {
      std::vector<RunResult> results;
      double max_el       = std::numeric_limits<double>::infinity();
      double mean         = std::numeric_limits<double>::infinity();
      double std          = std::numeric_limits<double>::infinity();
      double success_rate = std::numeric_limits<double>::infinity();
};

inline SuiteSummary summarize(const std::vector<RunResult> &rr, double tol = 1.0e-6)
{
   if (rr.size() == 0) return {};
   SuiteSummary res;
   res.mean         = 0.;
   res.std          = 0.;
   res.success_rate = 0.;

   res.max_el = 0.;

   res.results = rr;

   for (size_t i = 0; i < rr.size(); i++) {
      res.mean += rr[i].distance;
      res.std += rr[i].distance * rr[i].distance;
      if (rr[i].distance < tol) {
         res.success_rate += 1.;
      }
      if (rr[i].distance > res.max_el) res.max_el = rr[i].distance;
   }

   double s = (double)rr.size();
   double f = (rr.size() == 1 ? 1 : (s / (s - 1.)));

   res.mean /= s;
   res.std = sqrt((res.std / s) - res.mean * res.mean) * f;
   res.success_rate /= s;

   return res;
}

inline SuiteSummary run_all(nadir::Minimizer &minimizer, size_t runs_per_dim, double tol,
                            std::vector<size_t> dim = {50})
{

   if (dim.size() == 0) throw std::invalid_argument("run_all: Empty dimensions array");

   for (size_t i = 0; i < dim.size(); i++) {
      dim[i] = std::clamp(dim[i], 2ul, 100ul);
   }
   std::vector<RunResult> all;

   for (size_t i = 0; i < dim.size(); i++) {
      auto problems = make_suite(dim[i]);
      for (size_t j = 0; j < problems.size(); j++) {
         for (std::size_t r = 0; r < runs_per_dim; r++) {
            std::cout << "Running: " << problems[j].name << std::endl;
            all.emplace_back(run_one(minimizer, problems[j]));
         }
      }
   }
   auto S = summarize(all, tol);

   std::cout << "=== Nadir Bench Summary ===\n";
   std::cout << "Total runs: " << dim.size() * runs_per_dim << "\n";
   std::cout << "Tol: " << tol << ",  success-rate: " << S.success_rate * 100. << "%" << "\n";
   std::cout << "Mean: " << S.mean << ", Std: " << S.std << ", Max: " << S.max_el << "\n";

   for (auto &r : S.results) {
      std::cout << " - " << r.name << "  evals=" << r.evals << "  dist=" << r.distance << "\n";
   }

   return S;
}