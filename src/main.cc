#include <boost/make_shared.hpp>

#include <roboptim/core/linear-function.hh>
#include <roboptim/core/differentiable-function.hh>
#include <roboptim/core/problem.hh>
#include <roboptim/core/solver.hh>
#include <roboptim/core/solver-factory.hh>

#include <roboptim/trajectory/cubic-b-spline.hh>
#include <roboptim/trajectory/visualization/trajectory.hh>
// Define a solver type.
//
// Cost: Differentiable Function (gradient computations but no hessian)
// Constraint: Differentiable Function or Linear Function
typedef roboptim::Solver<
  roboptim::DifferentiableFunction,    // cost function
  boost::mpl::vector<
    roboptim::LinearFunction,          // constraint type 1
    roboptim::DifferentiableFunction   // constraint type 2
    >
  > solver_t;

class Cost : public roboptim::DifferentiableFunction
{
public:
  Cost ()
    // input size, output size, description
    : roboptim::DifferentiableFunction (10, 1, "the cost function")
  {}

  ~Cost ()
  {}

private:

  // result = f(x)
  virtual void impl_compute (result_t& result, const argument_t& /* x */) const
  {
    result[0] = 42.;
  }

  // grad = f.gradient (x)
  virtual void impl_gradient
  (gradient_t& grad, const argument_t& /* x */, size_type /* i */) const
  {
    // put all the elements to zero
    grad.setZero ();
  }
};

class Constraint : public roboptim::DifferentiableFunction
{
public:
  Constraint ()
    // input size, output size, description
    : roboptim::DifferentiableFunction (10, 1, "the constraint")
  {}

  ~Constraint ()
  {}

private:

  // result = f(x)
  virtual void impl_compute (result_t& result, const argument_t& x) const
  {
    result[0] = x[1];
  }

  // grad = f.gradient (x)
  virtual void impl_gradient
  (gradient_t& grad, const argument_t& /* x */, size_type /* i */) const
  {
    grad.setZero ();
    grad[1] = 1.;
  }
};

int main ()
{
  // create the cost function
  Cost cost;

  // evaluate cost function and associated gradient at a particular point x
  Cost::vector_t x (10);
  x << 0,1,2,3,4,5,6,7,8,9;
  std::cerr << cost(x) << std::endl;
  std::cerr << cost.gradient(x) << std::endl;

  // create the problem
  solver_t::problem_t problem (cost);

  // add the constraint
  boost::shared_ptr<roboptim::DifferentiableFunction> constraint =
    boost::make_shared<Constraint> ();

  // constraint: define associated interval
  solver_t::problem_t::intervals_t intervals (1);
  intervals[0] = 
    roboptim::Function::makeInterval (5, 10);
  // constraint: ...and scale
  solver_t::problem_t::scales_t scales (1);
  scales[0] = 1.;
  // constraint: add it to the problem
  problem.addConstraint (constraint, intervals, scales);

  // define the starting point
  problem.startingPoint () = x;

  // add bounds to one variable
  problem.argumentBounds ()[0] =
    roboptim::Function::makeInterval (0, 1);

  // creating the solver
  roboptim::SolverFactory<solver_t> factory ("cfsqp", problem);
  solver_t& solver = factory ();

  // display the solver and problem
  std::cerr << solver << std::endl;

  // solve the problem
  solver.solve ();
  
  // display the result
  std::cerr << solver.minimum () << std::endl;

  /*
  // create a spline
  roboptim::CubicBSpline::interval_t timeRange =
    roboptim::CubicBSpline::makeInterval (0., 10.);

  roboptim::CubicBSpline::vector_t parameters (2 * 10);
  parameters =
    roboptim::CubicBSpline::vector_t::Random (2 * 10);

  roboptim::CubicBSpline spline (timeRange, 2, parameters);

  // display trajectory as Gnuplot data
  roboptim::visualization::Gnuplot gnuplot =
    roboptim::visualization::Gnuplot::make_interactive_gnuplot ();
  gnuplot << roboptim::visualization::gnuplot::plot_xy (spline);
  std::cout << gnuplot;
  */
  return 0;
}
