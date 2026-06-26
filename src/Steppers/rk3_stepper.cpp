#include "Steppers/rk3_stepper.hpp"

using namespace diffeq;

void RK3Stepper::step(const RHS& f, double& t, Vec& y, double h) {
    if (k1.size() != y.size()) {
        k1.resize(y.size());
        k2.resize(y.size());
        k3.resize(y.size());
        y2.resize(y.size());
        y3.resize(y.size());
    }

    k1 = f(t, y);

    for (size_t i = 0; i < y.size(); ++i) {
        y2[i] = y[i] + 0.5 * h * k1[i];
    }

    k2 = f(t + 0.5 * h, y2);

    for (size_t i =0; i < y.size(); ++i) {
        y3[i] = y[i] - h * k1[i] + 2.0 * h * k2[i];
    }

    k3 = f(t + h, y3);

    for (size_t i = 0; i < y.size(); ++i) {
        y[i] += (h / 6.0) * (k1[i] + 4.0 * k2[i] + k3[i]);
    }

    t += h;
}