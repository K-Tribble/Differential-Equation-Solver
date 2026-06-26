#pragma once
#include "stepper.hpp"

namespace diffeq {

class RK3Stepper : public Stepper {
    public:
        void step(const RHS& f, double& t, Vec& y, double h) override;

        int GetOrder() const override {return order;};
        const char* GetName() const override {return name;};

    private:
        const char* name = "Runge-Kutta 3";
        const int order = 3;

        Vec k1;
        Vec k2;
        Vec k3;

        Vec y2;
        Vec y3;
};

}