#include "nadir/chain_minimizers.h"

namespace nadir
{
ChainedMinimizers::STATUS ChainedMinimizers::run()
{
   if (_minimizers.empty()) {
      throw std::runtime_error("Empty minimizers queue on run");
   }
   auto &current = _minimizers.front();

   _current_status     = current->minimize();
   _current_parameters = current->GetParameters();
   _minimizers.pop();

   if (!_minimizers.empty()) _minimizers.front()->SetInitialParameters(_current_parameters);

   return _current_status;
}

std::vector<ChainedMinimizers::STATUS> ChainedMinimizers::run_all()
{
   std::vector<STATUS> status;
   while (!_minimizers.empty()) {
      status.push_back(run());
   }
   return status;
}

void ChainedMinimizers::push_back(Minimizer *m)
{
   if (m == nullptr) throw std::invalid_argument("Nullptr given to ChainedMinimizer.push_back");
   _minimizers.push(m);
}

} // namespace nadir