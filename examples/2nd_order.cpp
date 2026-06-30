#include "solver.hpp"
#include "Steppers/rk4_stepper.hpp"
#include <cmath>
#include <iostream>

using namespace diffeq;

int main() {

    double c = 0.5; // Damping Coefficient
    double k = 1.0; // Spring Constant

    RHS f = [c, k](double t, const Vec& y) -> Vec {
        Vec dydt(2);
        dydt[0] = y[1];
        dydt[1] = -k * y[0] - c * y[1];
        return dydt;
    };

    Vec y0 = {1.0, 0.0};
    double t0 = 0.0;
    double t_end = 1;
    double h = 0.01;

    IVPProblem prob(f, y0, t0);

    RK4Stepper rk4;
    Solver solver(rk4);

    auto result = solver.integrateFixedSteps(prob, t_end, h);

    solver.printResults(result);

    return 0;
}