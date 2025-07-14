#include "nadir/shampoo.h"
#include <Eigen/Dense>

namespace
{

int inverse_matrix_sqrt(const Eigen::MatrixXd &A, double epsilon, Eigen::MatrixXd &res)
{
   Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(A);

   if (eigensolver.info() != Eigen::Success) {
      // Eigen decomposition failed
      return 1;
   }

   Eigen::VectorXd eigenvalues  = eigensolver.eigenvalues();
   Eigen::MatrixXd eigenvectors = eigensolver.eigenvectors();

   for (long int i = 0; i < eigenvalues.size(); i++) {
      if (eigenvalues(i) < -1.0e-10) {
         std::fprintf(stderr, "Warning: negative eigenvalue %ld detected in G! %.6e\n", i,
                      eigenvalues(i));
      }
   }

   // Compute Lambda^{-1/2}
   Eigen::VectorXd inv_sqrt_eigenvalues = (eigenvalues.array() + epsilon).inverse().sqrt();

   // Reconstruct G^{-1/2}
   Eigen::MatrixXd D_inv_sqrt = inv_sqrt_eigenvalues.asDiagonal();
   res                        = eigenvectors * D_inv_sqrt * eigenvectors.transpose();

   return 0;
}

} // namespace

namespace nadir
{

Shampoo::Shampoo(MetaParameters &mp, NadirCostFunction &fnc, Eigen::VectorXd pars)
    : Minimizer(fnc, pars), _mp(mp), _gt(Eigen::VectorXd::Zero(pars.size()))
{
   _G_acc  = mp.eps * Eigen::MatrixXd::Identity(pars.size(), pars.size());
   _P      = mp.eps * Eigen::MatrixXd::Identity(pars.size(), pars.size());
   _eps_Id = mp.eps * Eigen::MatrixXd::Identity(pars.size(), pars.size());
}

Shampoo::Shampoo(MetaParameters &mp, NadirCostFunction &fnc, long int n_par)
    : Minimizer(fnc, n_par), _mp(mp), _gt(Eigen::VectorXd::Zero(n_par))
{
   _G_acc  = mp.eps * Eigen::MatrixXd::Identity(n_par, n_par);
   _P      = mp.eps * Eigen::MatrixXd::Identity(n_par, n_par);
   _eps_Id = mp.eps * Eigen::MatrixXd::Identity(n_par, n_par);
}

Minimizer::STATUS Shampoo::minimize()
{
   Minimizer::STATUS status = Minimizer::STATUS::SUCCESS;

   double f_old = 0, f_new = 0.;
   _fnc.get().Evaluate(_parameters, f_old, _gt);

   for (size_t t = 1; t < _mp.max_it; t++) {

      _G_acc += _gt * _gt.transpose() + _eps_Id;
      _G_acc = 0.5 * (_G_acc + _G_acc.transpose());

      _G_acc = _mp.beta * _G_acc + (1 - _mp.beta) * (_gt * _gt.transpose());

      // Force symmetry
      _G_acc = 0.5 * (_G_acc + _G_acc.transpose());

      // Add epsilon for numerical stability
      _G_acc += _eps_Id;

      // Compute inverse sqrt of G
      int c = ::inverse_matrix_sqrt(_G_acc, _mp.eps, _P);
      if (c) {
         return Minimizer::STATUS::FAILURE;
      }

      _parameters -= _mp.lambda * (_P * _gt);

      _fnc.get().Evaluate(_parameters, f_new, _gt);

      if (_fnc_callback) {
         if (!_fnc_callback->get().Evaluate(_parameters)) {
            return STATUS::ABORT;
         }
      }

      double gradient_norm = sqrt(_gt.squaredNorm());
      if (gradient_norm < _mp.grad_toll) {
         status = STATUS::SUCCESS;
         break;
      }

      if (std::fabs(f_new - f_old) < _mp.diff_value_toll) {
         status = STATUS::LOW_DIFF;
         break;
      }

      _buffer << "- {Iteration: " << t;
      _buffer << ", Function value: " << f_new;
      _buffer << ", Gradient norm: " << gradient_norm << "}" << std::endl;

      f_old = f_new;
   }

   return status;
}

} // namespace nadir