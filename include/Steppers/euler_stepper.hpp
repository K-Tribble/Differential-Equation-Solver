#pragma once
#include "stepper.hpp"

namespace diffeq {

class EulerStepper : public Stepper{
    public:
        EulerStepper() = default;
        // step implements y_{n+1} = y_n + h * f(t_n, y_n)
        void step(const RHS& f, double& t, Vec& y, double h) override;

        int GetOrder() const override {return order;};
        const char* GetName() const override {return name;};

    private:
        // Vec dydt_; // reusable buffer to avoid allocations per step
        const char* name = "Euler";
        const int order = 1;

        Vec dydt;
};

}