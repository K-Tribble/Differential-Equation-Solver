#include <iostream>
#include <cmath>
#include <vector>
#include "pde/heat_equation.hpp"
#include "solver.hpp"
#include "Steppers/rk4_stepper.hpp"
#include "errorfunc/metric.hpp"

using namespace diffeq;
using namespace diffeq::pde;
using namespace diffeq::errorfunc;

// Example: 1D heat equation on x in [0,1], Dirichlet zero boundaries.
// Analytic solution for initial condition u(x,0) = sin(pi x) is
// u(x,t) = sin(pi x) * exp(-alpha * pi^2 * t).

int main() {
    const double L = 1.0;
    const std::size_t n_interior = 80; // interior points
    const double dx = L / (static_cast<double>(n_interior) + 1.0);
    const double alpha = 1.0;

    // Time integration parameters
    const double t_end = 0.1;
    // Choose dt safely small for explicit RK4 (stability ~ O(dx^2)).
    const double dt = 0.25 * dx * dx / alpha;

    BoundaryConditions1D bc;


    // Build RHS using Dirichlet zero BCs on both ends
    auto rhs = make_heat_rhs(alpha, n_interior, dx, bc);

    IVPProblem prob;
    prob.f = rhs;
    prob.t0 = 0.0;

    // Initial condition u(x,0) = sin(pi x) at interior points
    Vec y0(n_interior);
    for (std::size_t i = 0; i < n_interior; ++i) {
        double x = (i + 1) * dx; // interior node i corresponds to x = (i+1)*dx
        y0[i] = std::sin(M_PI * x);
    }
    prob.y0 = y0;

    // Exact solution function for the interior nodes
    auto exact = [n_interior, dx, alpha](double t) -> Vec {
        Vec v(n_interior);
        for (std::size_t i = 0; i < n_interior; ++i) {
            double x = (i + 1) * dx;
            v[i] = std::sin(M_PI * x) * std::exp(-alpha * M_PI * M_PI * t);
        }
        return v;
    };

    // Integrate with RK4
    RK4Stepper rk4;
    Solver solver(rk4);

    auto result = solver.integrateFixedSteps(prob, t_end, dt, exact, L2);

    result.print_info();

    std::cout << "Heat 1D example\n";
    std::cout << " n_interior = " << n_interior << " dx = " << dx << " dt = " << dt << "\n";
    std::cout << " Steps taken = " << result.n_steps << " total time = " << result.total_time << " s\n";
    std::cout << " Final error (" << result.error_func_name << ") = " << result.final_error << "\n";

    return 0;
}
