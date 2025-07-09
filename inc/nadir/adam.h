#ifndef NADIR_ADAM_H
#define NADIR_ADAM_H

#include <Eigen/Core>
#include <functional>
#include <iostream>
#include <sstream>
#include <fstream>

namespace nadir
{

class Adam
{
   public:
      enum class STATUS : unsigned {
         SUCCESS  = 0,
         ABORT    = 1,
         MAX_IT   = 2,
         LOW_DIFF = 3,
         CONTINUE = 4,
      };

      enum class ADAM_VARIANT : unsigned {
         CLASSIC,
         AMSGRAD,
         AMSGRAD_V2,
         NADAM,
         ADAMW,
         ADABELIEF,
         ADABELIEF_W, // AdaBelief with decoupled weight decay
         LION,
         RADAM,
      };

      struct MetaParameters {
            ADAM_VARIANT variant   = ADAM_VARIANT::CLASSIC;
            size_t max_it          = 100;
            double alpha           = 0.001;
            double beta1           = 0.9;
            double beta2           = 0.999;
            double eps             = 1.0e-8;
            double grad_toll       = 1.0e-8;
            double diff_value_toll = 1.0e-8;
            double lambda          = 0.; // Only used for the AdamW variant
      };

      Adam(std::function<void(const Eigen::VectorXd &, double &, Eigen::VectorXd &)> fnc,
           Eigen::VectorXd pars = Eigen::VectorXd::Zero(1));

      Adam(std::function<void(const Eigen::VectorXd &, double &, Eigen::VectorXd &)> fnc,
           long int n_par = 1);

      void SetScheduler(const std::function<double(size_t)> &scheduler)
      {
         _scheduler = scheduler;
      }

      void SetInitialParameters(const Eigen::VectorXd &pars)
      {
         _parameters = pars;
      }
      void AddCallBack(const std::function<bool(Eigen::VectorXd &)> &f)
      {
         _have_callback = true;
         _fnc_callback  = f;
      }

      const Eigen::VectorXd &GetParameters() const
      {
         return _parameters;
      }

      void SetMetaParameters(const MetaParameters &mp)
      {
         _mp = mp;
      }
      const MetaParameters &GetMetaParameters() const
      {
         return _mp;
      }

      STATUS minimize();

      void FlusToFile(const std::string &filename) const
      {
         std::ofstream file(filename);
         if (file.is_open()) {
            file << _buffer.str();
            file.close();
         } else {
            throw std::runtime_error("FlushToFile: Could not open file: " + filename);
         }
      }

      void FlusToStdout() const
      {
         std::cout << _buffer.str() << std::endl;
      }

   private:
      std::function<void(const Eigen::VectorXd &, double &, Eigen::VectorXd &)> _fnc;
      bool _have_callback;
      std::function<bool(Eigen::VectorXd &)> _fnc_callback;
      Eigen::VectorXd _parameters;
      double f_new, f_old;
      Eigen::VectorXd _gt, _mt, _vt;
      Eigen::VectorXd _vt_hat; // Used only for AMS_grad
      MetaParameters _mp;
      std::function<double(size_t)> _scheduler;

      std::ostringstream _buffer;

      size_t step(size_t);
      size_t step_ams(size_t);
      size_t step_ams_v2(size_t);
      size_t step_nadam(size_t);
      size_t step_adamw(size_t);
      size_t step_adabelief(size_t);
      size_t step_adabelief_w(size_t);
      size_t step_lion(size_t);
      size_t step_radam(size_t);

      const std::function<double(double)> sign = [](double x) {
         return (x > 0) - (x < 0);
      };
};
} // namespace nadir

#endif