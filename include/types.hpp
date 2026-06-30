#pragma once
#include <vector>
#include <functional>

namespace diffeq {

    using Vec = std::vector<double>;
    using RHS = std::function<Vec(double t, const Vec& y)>;

    namespace pde {
        // Simple boundary condition types
        enum class BCType { Dirichlet, Neumann, Periodic };

        // Boundary value / flux signature: value at a boundary as a function of time
        using BCFunc = std::function<double(double)>;
    }

    /*
    Later can add:

    Matrix alias (for Jacobians)

    Real type template (float/double/long double)

    Parallel vector implementations (Eigen, custom allocators, etc.)
    */
}