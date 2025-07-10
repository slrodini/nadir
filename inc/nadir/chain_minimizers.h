#ifndef NADIR_CHAIN_MINIMIZERS_H
#define NADIR_CHAIN_MINIMIZERS_H

#include "nadir/abstract_classes.h"
#include <queue>

namespace nadir
{

class ChainedMinimizers
{
   public:
      using STATUS = Minimizer::STATUS;

   public:
      ChainedMinimizers() = default;

      void push_back(Minimizer *m);

      STATUS run();

      std::vector<STATUS> run_all();

   private:
      std::queue<Minimizer *> _minimizers;
      Eigen::VectorXd _current_parameters;
      STATUS _current_status;
};

} // namespace nadir
#endif // NADIR_CHAIN_MINIMIZER_H