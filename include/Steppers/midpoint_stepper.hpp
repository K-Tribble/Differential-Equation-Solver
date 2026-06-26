#pragma once
#include "stepper.hpp"

// Midpoint stepper, also 2nd order RK

namespace diffeq {

class MidpointStepper : public Stepper {
    public: 
        void step(const RHS& f, double& t, Vec& y, double h) override;

        int GetOrder() const override {return order;};
        const char* GetName() const override {return name;};

    private:
        const char* name = "Midpoint";
        const int order = 2;

        Vec k1;
        Vec k2;

        Vec y_mid;
};
}