#include "nadir/simulated_annealing.h"
#include "./ran2.cpp"

namespace nadir
{
// =================================================================================================
// General functions
// =================================================================================================
void set_seed(long int seed)
{
   if (::seeded) throw std::runtime_error("Cannot seed random engine more than once.");
   ::seeded = true;
   ::_idum  = seed;
}

// =================================================================================================
// ITx implementations
// =================================================================================================
std::pair<double, double> IT1::operator()()
{
   return {_k, _Tf};
}

std::pair<double, double> IT2::operator()()
{
   double res = 0;
   _fnc.Evaluate(_par, res);
   return {_k * res, _Tf};
}

void itx_random_walk(NadirCostFunction &fnc, Eigen::VectorXd &pars, size_t n, double step_size,
                     std::vector<double> &out)
{

   for (size_t i = 0; i < n; i++) {
      double s = 0.;
      for (long int i = 0; i < pars.size(); i++) {
         pars(i) += step_size * (2. * ::_random_uniform() - 1.);
      }
      fnc.Evaluate(pars, s);
      out.push_back(s);
   }
}

std::pair<double, double> IT3::operator()()
{
   std::vector<double> rw;
   itx_random_walk(_fnc, _par, _n, _step_size, rw);

   double m = 0.;
   double l = std::numeric_limits<double>::max();

   for (size_t i = 0; i < rw.size() - 1; i++) {
      double delta_iip1 = std::fabs(rw[i] - rw[i + 1]);

      m = std::max(m, delta_iip1);
      l = std::min(l, delta_iip1);
   }
   return {m * _k, l * _k};
}

std::pair<double, double> IT4::operator()()
{
   std::vector<double> rw;
   itx_random_walk(_fnc, _par, _n, _step_size, rw);

   double average = 0;

   for (size_t i = 0; i < rw.size() - 1; i++) {
      average += std::fabs(rw[i] - rw[i + 1]);
   }
   return {average * _k / static_cast<double>(_n), _Tf};
}

std::pair<double, double> IT6::operator()()
{
   std::vector<double> rw;
   itx_random_walk(_fnc, _par, _n, _step_size, rw);

   double average = 0;
   double N       = 0;

   // Note: the algorithm seems to be written for the true average,
   // not the average of the absolute values, but I think the intended
   // behavior should be with the abs of the gap, otherwise there is
   // a good probability that the average gap is close to zero...
   // Not sure, for the moment it is implemented with the absolute value
   for (size_t i = 0; i < rw.size(); i++) {
      for (size_t j = i + 1; j < rw.size(); j++) {
         average += std::fabs(rw[i] - rw[j]);
         N++;
      }
   }
   return {std::fabs(average * _k / N / log(_p0)), _Tf};
}

std::pair<double, double> IT7::operator()()
{
   std::vector<double> rw;
   itx_random_walk(_fnc, _par, _n, _step_size, rw);

   double average = 0;
   double N       = 0;

   double m = 0.;
   double l = std::numeric_limits<double>::max();

   // Note: same as previous
   for (size_t i = 0; i < rw.size(); i++) {
      for (size_t j = i + 1; j < rw.size(); j++) {
         double delta_ij = std::fabs(rw[i] - rw[j]);

         average += delta_ij;
         m = std::max(m, delta_ij);
         l = std::min(l, delta_ij);
         N++;
      }
   }
   average /= N;
   double d1 = 1. - _l1 - _l1p;
   double d2 = 1. - _l2 - _l2p;
   double T0 = _k * (d1 * l + _l1 * average + _l1p * m);
   double Tf = _k * (d2 * l + _l2 * average + _l2p * m);

   return {T0, Tf};
}

// =================================================================================================
// SCx implementations
// =================================================================================================

bool SC2::operator()(const SimAnnContext &context)
{
   return context.moves > _n;
}

bool SC3::operator()(const SimAnnContext &context)
{
   return context.T < context.Tf;
}

bool SC4::operator()(const SimAnnContext &context)
{
   return context.cooling_steps > _n;
}

bool SC6::operator()(const SimAnnContext &context)
{
   return context.failed_candidate_moves > _n;
}

bool SC7::operator()(const SimAnnContext &context)
{
   return context.acceptance_rate < _ar;
}

// =================================================================================================
// NEx implementations
// =================================================================================================

void NE3::operator()(SimAnnContext &context, NadirCostFunction &fnc,
                     const Eigen::VectorXd &_incumbent_pars, Eigen::VectorXd &_pars)
{
   // Randomly generated gaussian around current incumbent
   for (long int i = 0; i < _pars.size(); i++) {
      _pars(i) = _incumbent_pars(i) + _width * ::_random_normal();
   }
   double curr_inc = 0;
   fnc.Evaluate(_pars, curr_inc);

   double best_inc = curr_inc;
   _best           = _pars;

   for (size_t k = 1; k < _n; k++) {
      // Randomly generated gaussian around current incumbent
      for (long int i = 0; i < _pars.size(); i++) {
         _pars(i) = _incumbent_pars(i) + _width * ::_random_normal();
      }
      fnc.Evaluate(_pars, curr_inc);

      if (curr_inc <= best_inc) {
         best_inc = curr_inc;
         _best    = _pars;
      }
   }
   _pars = _best;

   context.proposed_sol = best_inc;
}

void NE4::operator()(SimAnnContext &context, NadirCostFunction &fnc,
                     const Eigen::VectorXd &_incumbent_pars, Eigen::VectorXd &_pars)
{
   // Randomly generated gaussian around current incumbent
   for (long int i = 0; i < _pars.size(); i++) {
      _pars(i) = _incumbent_pars(i) + _width * ::_random_normal();
   }
   double curr_inc = 0;
   fnc.Evaluate(_pars, curr_inc);

   // Early termination if we find improvement w.r.t. current incumbent
   if (curr_inc <= context.incumbent_sol) {
      context.proposed_sol = curr_inc;
      return;
   }

   double best_inc = curr_inc;
   _best           = _pars;

   for (size_t k = 1; k < _n; k++) {
      // Randomly generated gaussian around current incumbent
      for (long int i = 0; i < _pars.size(); i++) {
         _pars(i) = _incumbent_pars(i) + _width * ::_random_normal();
      }
      fnc.Evaluate(_pars, curr_inc);

      if (curr_inc <= best_inc) {
         best_inc = curr_inc;
         _best    = _pars;
      }

      // Early termination if we find improvement w.r.t. current incumbent
      if (best_inc < context.incumbent_sol) {
         _pars                = _best;
         context.proposed_sol = best_inc;
         return;
      }
   }
   _pars                = _best;
   context.proposed_sol = best_inc;
}

void NE5::operator()(SimAnnContext &context, NadirCostFunction &fnc,
                     const Eigen::VectorXd &_incumbent_pars, Eigen::VectorXd &_pars)
{
   // Randomly generated gaussian around current incumbent
   long int i = static_cast<long int>(std::floor(_pars.size() * ::_random_uniform()));
   _pars(i)   = _incumbent_pars(i) + _width * ::_random_normal();

   double curr_inc = 0;
   fnc.Evaluate(_pars, curr_inc);

   double best_inc = curr_inc;
   _best           = _pars;

   for (size_t k = 1; k < _n; k++) {
      // Randomly generated gaussian around current incumbent

      i = static_cast<long int>(std::floor(_pars.size() * ::_random_uniform()));

      _pars(i) = _incumbent_pars(i) + _width * ::_random_normal();

      fnc.Evaluate(_pars, curr_inc);

      if (curr_inc <= best_inc) {
         best_inc = curr_inc;
         _best    = _pars;
      }
   }
   _pars                = _best;
   context.proposed_sol = best_inc;
}

void NE6::operator()(SimAnnContext &context, NadirCostFunction &fnc,
                     const Eigen::VectorXd &_incumbent_pars, Eigen::VectorXd &_pars)
{
   // Randomly generated gaussian around current incumbent
   long int i = static_cast<long int>(std::floor(_pars.size() * ::_random_uniform()));
   _pars(i)   = _incumbent_pars(i) + _width * ::_random_normal();

   double curr_inc = 0;
   fnc.Evaluate(_pars, curr_inc);

   // Early termination if we find improvement w.r.t. current incumbent
   if (curr_inc <= context.incumbent_sol) {
      context.proposed_sol = curr_inc;
      return;
   }
   double best_inc = curr_inc;
   _best           = _pars;

   for (size_t k = 1; k < _n; k++) {
      // Randomly generated gaussian around current incumbent
      i = static_cast<long int>(std::floor(_pars.size() * ::_random_uniform()));

      _pars(i) = _incumbent_pars(i) + _width * ::_random_normal();

      fnc.Evaluate(_pars, curr_inc);

      if (curr_inc <= best_inc) {
         best_inc = curr_inc;
         _best    = _pars;
      }

      // Early termination if we find improvement w.r.t. current incumbent
      if (best_inc < context.incumbent_sol) {
         _pars                = _best;
         context.proposed_sol = best_inc;
         return;
      }
   }
   _pars                = _best;
   context.proposed_sol = best_inc;
}

// =================================================================================================
// ACx implementations
// =================================================================================================

bool AC1::operator()(const SimAnnContext &context)
{
   bool r = ::_random_uniform();
   return (context.proposed_sol <= context.incumbent_sol) ||
          r < exp(-(context.proposed_sol - context.incumbent_sol) / context.T);
}

bool AC2::operator()(const SimAnnContext &context)
{
   // Early reject if the exponential would be too small
   if ((context.proposed_sol - context.incumbent_sol) > -context.T * _log_eps) return false;
   bool r = ::_random_uniform();
   return (context.proposed_sol <= context.incumbent_sol) ||
          r < exp(-(context.proposed_sol - context.incumbent_sol) / context.T);
}

bool AC3::operator()(const SimAnnContext &context)
{
   if ((context.proposed_sol < context.incumbent_sol)) return true;
   double fs  = _quality(context.incumbent_sol);
   double fsp = _quality(context.proposed_sol);
   if (fs < fsp && fsp <= fs * _phi) {
      bool r = ::_random_uniform();
      return r < exp(-(context.proposed_sol - context.incumbent_sol) / context.T);
   } else return false;
}

bool AC4::operator()(const SimAnnContext &context)
{
   if (context.proposed_sol <= context.incumbent_sol) return true;
   double p = exp(-_beta(context.T) * pow(_quality(context.proposed_sol), _g) *
                  (context.proposed_sol - context.incumbent_sol));
   return ::_random_uniform() < p;
}

bool AC5::operator()(const SimAnnContext &context)
{
   if (context.proposed_sol <= context.incumbent_sol) {
      _k++;
      return true;
   }
   bool r   = ::_random_uniform();
   double p = _p0 * pow_n(_r, _k - 1);
   if (r < p) {
      _k++;
      return true;
   }
   return false;
}

bool AC6::operator()(const SimAnnContext &context)
{
   return (context.proposed_sol - context.incumbent_sol) <= _phi(context.T);
}

bool AC7::operator()(const SimAnnContext &context)
{
   return context.proposed_sol <= _phi(context.T, _k++);
}

bool AC10::operator()(const SimAnnContext &context)
{
   return context.proposed_sol <= context.incumbent_sol;
}

// =================================================================================================
// CSx implementations
// =================================================================================================

void CS1::operator()(SimAnnContext &context)
{
   context.T = _alpha * pow(context.T, _beta);
}

void CS2::operator()(SimAnnContext &context)
{
   context.T = _alpha * context.T;
}

void CS3::operator()(SimAnnContext &context)
{
   context.T = _alpha / log(_beta + static_cast<double>(context.cooling_steps));
}

void CS4::operator()(SimAnnContext &context)
{
   context.T = _alpha / (_beta + log(static_cast<double>(context.cooling_steps)));
}

void CS5::operator()(SimAnnContext &context)
{
   double den = _alpha + _beta * context.T;
   if (den <= context.T) {
      throw std::invalid_argument("CS5: alpha and beta must be such that alpha + beta T > T");
   }

   context.T = context.T / den;
}

void CS7::operator()(SimAnnContext &context)
{
   double den = 1. + _beta * context.T;
   if (den <= 0) {
      throw std::invalid_argument("CS7: 1 + beta T must always be >0");
   }
   double res = _alpha / den;

   if (res > context.T) {
      throw std::invalid_argument("CS7: a/(1 + beta T) must always be <=T");
   }
   context.T = res;
}

void CS9::operator()(SimAnnContext &context)
{
   context.T = std::max(context.T - _a, 0.);
}

void CS10::operator()(SimAnnContext &context)
{
   context.T = _Tx;
}

void CS11::operator()(SimAnnContext &context)
{
   context.T = _Tx + _a * ::_random_uniform();
}

// =================================================================================================
// TLx implementations
// =================================================================================================

size_t TL1::operator()(const SimAnnContext &)
{
   return _k;
}

size_t TL2::operator()(const SimAnnContext &)
{
   return _k * static_cast<size_t>(std::ceil(_width));
}

size_t TL3::operator()(const SimAnnContext &)
{
   return _k * static_cast<size_t>(std::ceil(_width));
}

size_t TL4::operator()(const SimAnnContext &)
{
   return _k;
}

size_t TL5::operator()(const SimAnnContext &)
{
   return _k;
}

size_t TL8::operator()(const SimAnnContext &context)
{
   return context.cooling_steps + _k;
}

size_t TL9::operator()(const SimAnnContext &context)
{
   return context.cooling_steps * _k;
}

size_t TL10::operator()(const SimAnnContext &context)
{
   return std::max(_k / context.cooling_steps, 1ul);
}

size_t TL11::operator()(const SimAnnContext &context)
{
   return static_cast<size_t>(std::ceil(pow(context.cooling_steps, 1 / _a)));
}

// =================================================================================================
// Annealing implementation
// =================================================================================================

SimAnnealing::STATUS SimAnnealing::minimize()
{
   STATUS status = STATUS::SUCCESS;
   auto [T0, Tf] = _itx();

   double tmp_sol;
   _fnc.get().Evaluate(_parameters, tmp_sol);

   SimAnnContext context{
       .iter                   = 0,
       .moves                  = 0,
       .T                      = T0,
       .cooling_steps          = 0,
       .failed_candidate_moves = 0,
       .acceptance_rate        = 1,
       .incumbent_sol          = tmp_sol,
       .best_sol               = tmp_sol,
       .proposed_sol           = tmp_sol,
       .T0                     = T0,
       .Tf                     = Tf,
   };

   double tot_generation = 0;
   double accepted       = 0;

   do {

      context.failed_candidate_moves = 0;

      size_t n_temp = _tlx(context);

      for (size_t i = 0; i < n_temp; i++) {
         context.iter++;

         // Generates the proposed solution in the neighbour
         _nex(context, _fnc, _incumbent_pars, _current_pars);
         tot_generation++;
         context.moves++;

         if (_acx(context)) {
            accepted++;
            // Accept the proposed solution
            context.incumbent_sol = context.proposed_sol;
            _incumbent_pars       = _current_pars;

            _buffer << "- {Iteration: " << context.iter
                    << ", Function value: " << context.proposed_sol;
            _buffer << ", temperature: " << context.T << "}" << std::endl;

            if (context.incumbent_sol <= context.best_sol) {
               context.best_sol = context.incumbent_sol;
               _parameters      = _incumbent_pars;

               _buffer << "- {Iteration: " << context.iter
                       << ", Function value: " << context.best_sol;
               _buffer << "}" << std::endl;

               // Note: invoke callback only on update of best solution
               if (_fnc_callback) {
                  _fnc_callback->get().Evaluate(_parameters);
               }
            }
         } else {
            context.failed_candidate_moves++;
         }
         context.acceptance_rate = accepted / tot_generation;
      }
      _csx(context);
      context.cooling_steps++;

   } while (!_scx(context));

   return status;
}

} // namespace nadir