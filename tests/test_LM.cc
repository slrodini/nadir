#include "nadir/nadir.h"
#include <ceres/ceres.h>

class CF : public nadir::NadirLevMarCostFunction, public ceres::CostFunction
{
   public:
      CF(long int n)
      {
         data = Eigen::VectorXd::Zero(n);
         for (long int i = 0; i < n; i++) {
            double dx = static_cast<double>(i) / static_cast<double>(n - 1);
            double x  = -3. + 6. * dx;
            data(i)   = exp(-x * x);
         }
         set_num_residuals(n);
         mutable_parameter_block_sizes()->push_back(1);
         mutable_parameter_block_sizes()->push_back(1);
         _p.push_back(new double);
         _p.push_back(new double);
      }

      void Evaluate(const Eigen::VectorXd &parameters, Eigen::VectorXd &residual,
                    Eigen::Block<Eigen::MatrixXd> *jacobian) override
      {
         double mu    = parameters(0);
         double sigma = parameters(1);

         auto gauss = [mu, sigma](double x) {
            return exp(-(x - mu) * (x - mu) / (2.0 * sigma * sigma));
         };

         for (long int i = 0; i < data.size(); i++) {
            double dx   = static_cast<double>(i) / static_cast<double>(data.size() - 1);
            double x    = -3. + 6. * dx;
            double th   = gauss(x);
            residual(i) = th - data(i);
            if (jacobian != nullptr) {
               (*jacobian)(i, 0) = th * (x - mu) / (sigma * sigma);
               (*jacobian)(i, 1) = th * (x - mu) * (x - mu) / (sigma * sigma * sigma);
            }
         }
      }

      bool Evaluate(double const *const *parameters, double *residuals,
                    double **jacobians) const override
      {
         double mu    = parameters[0][0];
         double sigma = parameters[1][0];

         auto gauss = [mu, sigma](double x) {
            return exp(-(x - mu) * (x - mu) / (2.0 * sigma * sigma));
         };

         for (long int i = 0; i < data.size(); i++) {
            double dx    = static_cast<double>(i) / static_cast<double>(data.size() - 1);
            double x     = -3. + 6. * dx;
            double th    = gauss(x);
            residuals[i] = th - data(i);
            if (jacobians != nullptr) {
               jacobians[0][i] = th * (x - mu) / (sigma * sigma);
               jacobians[1][i] = th * (x - mu) * (x - mu) / (sigma * sigma * sigma);
            }
         }
         return true;
      }

      long int ResidualNumber() override
      {
         return data.size();
      }

   public:
      Eigen::VectorXd data;
      std::vector<double *> _p;
};

int main()
{

   nadir::LevMarMinimizer::MetaParameters mp = nadir::LevMarMinimizer::MetaParameters{
       .grad_toll          = 0.,
       .diff_value_toll    = 1.0e-20,
       .eta1               = 0.75,
       .eta2               = 0.25,
       .real_time_progress = true,
   };

   CF cf(12);
   Eigen::VectorXd p(2);
   p(0) = -2;
   p(1) = 0.1;
   nadir::LevMarMinimizer lm(mp, cf, p);

   std::cout << nadir::LevMarMinimizer::print_status(lm.minimize()) << std::endl;
   std::cout << lm.GetParameters() << std::endl;

   CF *cf_cer       = new CF(12);
   *(cf_cer->_p[0]) = p(0);
   *(cf_cer->_p[1]) = p(1);
   ceres::Problem _problem;
   _problem.AddResidualBlock(cf_cer, nullptr, cf_cer->_p);
   ceres::Solver::Options _options;
   _options.minimizer_progress_to_stdout = true;
   _options.function_tolerance           = 0.;
   _options.gradient_tolerance           = 0.;
   _options.parameter_tolerance          = 0.;
   ceres::Solver::Summary _summary;
   ceres::Solve(_options, &_problem, &_summary);

   std::cout << *(cf_cer->_p[0]) << std::endl;
   std::cout << *(cf_cer->_p[1]) << std::endl;

   // Eigen::MatrixXd TB(2, 2);
   // TB(0, 0) = 1;
   // TB(0, 1) = 2;
   // TB(1, 0) = 3;
   // TB(1, 1) = 4;

   // Eigen::MatrixXd JtJ = Eigen::MatrixXd::Zero(2, 2);

   // JtJ.selfadjointView<Eigen::Lower>().rankUpdate(TB.transpose());

   // Eigen::VectorXd diag = (TB.array().square().colwise().sum()).sqrt();
   // std::cout << JtJ << std::endl;
   // std::cout << TB.transpose() * TB << std::endl;
   return 0;
}