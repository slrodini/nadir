#include "nadir/abstract_classes.h"

namespace nadir
{
class Lamb : public Minimizer
{
   public:
      /// Metaparameters for the Lamb minimizer
      struct MetaParameters {
            /// Max number of iteration
            size_t max_it = 100;
            /// Learning rate
            double alpha = 0.001;
            /// Exponentaial decay rate for the moving average of the 1st moment
            double beta1 = 0.9;
            /// Exponentaial decay rate for the moving average of the 2st moment
            double beta2 = 0.999;
            /// Regulator of the denominator in the update
            double eps = 1.0e-8;
            /// Tollerance on the gradient norm
            double grad_toll = 1.0e-8;
            /// Tollerance on the \f$ \Delta F \f$  in subsequent steps
            double diff_value_toll = 1.0e-8;
            /// Parameter for the decoupled weight decay variants
            double lambda = 0.;
            /// Real time progress to stderr
            bool real_time_progress = false;
            /// Parameters blocks
            std::vector<long int> par_blocks = {0};
            /// Scalin function
            std::function<double(double)> phi = [](double x) {
               return x;
            };
      };

      /// \name Constructor
      ///@{

      /**
       * @brief Construct a new Adam minimizer
       *
       * @param fnc  The cost function
       * @param pars The array of initial parameters
       */
      Lamb(const MetaParameters &mp, NadirCostFunction &fnc, Eigen::VectorXd pars);

      virtual ~Lamb() = default;
      ///@}

      /**
       * @brief Set the Scheduler object for variable learning rate
       *
       * @param scheduler The sceduler, as a function of the step count
       */
      void SetScheduler(const std::function<double(size_t)> &scheduler)
      {
         _scheduler = scheduler;
      }

      void ChangeMetaParameters(MetaParameters mp)
      {
         _mp = mp;
         throw std::runtime_error("lamb: MetaParameters need checks on par blocks");
      }

      /// Returns the meta-parameters
      MetaParameters GetMetaParameters() const
      {
         return _mp;
      }
      /**
       * @brief Main function: execute the minimization
       *
       * @return STATUS
       */
      STATUS minimize() override;

   private:
      /// The meta-parameters of the minimizer
      MetaParameters _mp;

      /// The scheduler for a time-dependent learning rate (defaulted to unit function)
      std::function<double(size_t)> _scheduler;

      void _reset() override
      {
         _scheduler = [](size_t) {
            return 1.;
         };
      }
};

} // namespace nadir