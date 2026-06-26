#include "Steppers/euler_stepper.hpp"
#include <cstddef>

using namespace diffeq;

void EulerStepper::step(const RHS& f, double& t, Vec& y, double h) {
    if (dydt.size() != y.size()) {
        dydt.resize(y.size());
    }

    dydt = f(t, y);
    for (std::size_t i = 0; i < y.size(); ++i) {
        y[i] += h * dydt[i];
    }

    t += h;
}