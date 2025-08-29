#include "bench_suite.h"

int main()
{
   nadir::NadirCostFunction dummy;
   Eigen::VectorXd p_dummy = Eigen::VectorXd::Zero(1);
   std::string div(80, '=');

   // ---------------------------------------------------------------------------------------------
   // DiffEvolution
   // ---------------------------------------------------------------------------------------------
   if (false) {
      std::cout << div << "\n";
      std::cout << "Running DiffEvolution...\n";

      nadir::DiffEvolution::MetaParameters mp = {
          .NP                 = 100,
          .CR                 = 0.9,
          .F                  = 0.8,
          .width              = 1.,
          .max_iter           = 10000,
          .real_time_progress = false,
      };
      nadir::DiffEvolution diffevo(mp, dummy, p_dummy);
      (void)run_all(diffevo, 5, 1.0e-12, {2, 4, 8, 12});
   }

   // ---------------------------------------------------------------------------------------------
   // jSOa
   // ---------------------------------------------------------------------------------------------
   if (false) {
      std::cout << div << "\n";
      std::cout << "Running jSOa...\n";

      nadir::jSOa::MetaParameters mp = {
          .NP_ini                = 100,
          .NP_min                = 20,
          .p_max                 = 0.25,
          .p_min                 = 0.125,
          .impr_archive_eviction = true,
          .ap                    = 0.2,
          .CR_term               = 0.1,
          .H                     = 5,
          .width                 = 1.,
          .max_iter              = 3000,
          .real_time_progress    = false,
      };

      nadir::jSOa jsoa(mp, dummy, p_dummy);
      (void)run_all(jsoa, 5, 1.0e-12, {2, 4, 8, 12});
   }

   // ---------------------------------------------------------------------------------------------
   // CMA-ES
   // ---------------------------------------------------------------------------------------------
   if (true) {
      std::cout << div << "\n";
      std::cout << "Running CMA-ES...\n";

      nadir::CMA_ES::MetaParameters mp = {
          .sigma              = 1.0,
          .max_fnc_eval       = 100000,
          .max_iter           = 200,
          .real_time_progress = false,
      };

      nadir::CMA_ES::BIPOP_MetaParameters mp_2 = {
          .sigma_ref          = 10.0,
          .max_fnc_eval       = 10000000,
          .max_iter           = 50,
          .real_time_progress = false,
      };
      nadir::CMA_ES cmaes(mp, mp_2, dummy, p_dummy);
      (void)run_all(cmaes, 5, 1.0e-8, {2, 4, 8, 12});
   }

   // ---------------------------------------------------------------------------------------------
   // SimulatedAnnealing
   // ---------------------------------------------------------------------------------------------
   if (false) {
      std::cout << div << "\n";
      std::cout << "Running SimulatedAnnealing...\n";

      // Fixed initial and final T
      nadir::IT1 itx(100., 1.0e-4);
      // Until min tempterature
      nadir::SC3 scx;
      // Best of k neighbour
      nadir::NE3 nex(0.5, 30);
      // Best sol
      nadir::AC10 acx;
      // Geometric cooling
      nadir::CS2 csx(0.99);
      // Fix Temperature length
      nadir::TL1 tlx(500);

      nadir::SimAnnealing annealing(dummy, p_dummy, itx, scx, nex, acx, csx, tlx, true, false);

      // (void)run_all(annealing, 5, 1.0e-12, {2, 4, 8, 12});
      RunResult res = run_one(annealing, make_rosenbrock(2));
      std::cout << div << "\n";
      std::cout << res.distance << std::endl;
   }

   // ---------------------------------------------------------------------------------------------
   // Adam
   // ---------------------------------------------------------------------------------------------
   if (false) {
      std::cout << div << "\n";
      std::cout << "Running adam...\n";

      nadir::Adam::MetaParameters mp = {
          .variant            = nadir::Adam::ADAM_VARIANT::ADABELIEF,
          .max_it             = 100000,
          .alpha              = 0.6,
          .beta1              = 0.9,
          .beta2              = 0.999,
          .eps                = 1.0e-4,
          .grad_toll          = 1.0e-16,
          .diff_value_toll    = 1.0e-16,
          .lambda             = 0.,
          .real_time_progress = false,
      };

      nadir::Adam adam(mp, dummy, p_dummy);
      auto linSched = [&mp](size_t t) {
         double tp        = static_cast<double>(t);
         double Tp        = static_cast<double>(mp.max_it);
         double alpha_max = 0.6;
         double alpha_min = 1.0e-5;
         return alpha_min + (alpha_max - alpha_min) * (1. - tp / Tp);
      };
      adam.SetScheduler(linSched);

      (void)run_all(adam, 5, 1.0e-12, {2, 4, 8, 12});
      // RunResult res = run_one(adam, make_rosenbrock(2));
      // std::cout << div << "\n";
      // std::cout << res.distance << std::endl;
   }

   // ---------------------------------------------------------------------------------------------
   // SOAA
   // ---------------------------------------------------------------------------------------------
   if (false) {
      std::cout << div << "\n";
      std::cout << "Running SOAA...\n";

      nadir::SOAA::MetaParameters mp = {
          .max_it             = 100000,
          .alpha              = 0.6,
          .gamma              = 0.1,
          .eps                = 1.0e-8,
          .grad_toll          = 1.0e-14,
          .diff_value_toll    = 1.0e-14,
          .real_time_progress = false,
      };

      nadir::SOAA soaa(mp, dummy, p_dummy);

      (void)run_all(soaa, 5, 1.0e-12, {2, 4, 8, 12});
   }

   // ---------------------------------------------------------------------------------------------
   // Shampoo
   // ---------------------------------------------------------------------------------------------
   if (false) {
      std::cout << div << "\n";
      std::cout << "Running Shampoo...\n";

      nadir::Shampoo::MetaParameters mp = {
          .max_it             = 2000,
          .beta               = 0.5,
          .lambda             = 0.01,
          .eps                = 1.0e-8,
          .grad_toll          = 1.0e-16,
          .diff_value_toll    = 1.0e-16,
          .real_time_progress = false,
      };

      nadir::Shampoo shampoo(mp, dummy, p_dummy);

      (void)run_all(shampoo, 5, 1.0e-12, {2, 4, 8, 12});
   }
}