#pragma once
#include "types.hpp"
#include <functional>
#include <cstddef>

namespace diffeq {
namespace pde {

struct BCSide {
    BCType type = BCType::Dirichlet;
    BCFunc value = BCFunc();

    BCSide() = default;

    BCSide(BCType bcType, BCFunc val) 
        : type(bcType), value(std::move(val)) {}
};

struct BoundaryConditions1D {
    BCSide left;
    BCSide right;

    BoundaryConditions1D() = default;

    BoundaryConditions1D(BCSide l, BCSide r) 
        : left(std::move(l)), right(std::move(r)){}

    // Constructor to initialize all boundaries the same way.
    BoundaryConditions1D(BCType t, BCFunc val) {
        BCSide side(t, std::move(val));

        left = side;
        right = std::move(side);
    }
};

struct BoundaryConditions2D {
    BCSide left;
    BCSide right;
    BCSide top;
    BCSide bottom;

    BoundaryConditions2D() = default;

    BoundaryConditions2D(BCSide l, BCSide r, BCSide t, BCSide b) 
        : left(std::move(l)), right(std::move(r)), top(std::move(t)), bottom(std::move(b)){}

    
    // Constructor to initialize all boundaries the same way.
    BoundaryConditions2D(BCType t, BCFunc val) {
        BCSide side(t, std::move(val));

        left = side;
        right = side;
        top = side;
        bottom = std::move(side);
    }
};

struct BoundaryConditions3D {
    BCSide left;
    BCSide right;
    BCSide top;
    BCSide bottom;
    BCSide front;
    BCSide back;

    BoundaryConditions3D() = default;

    BoundaryConditions3D(BCSide l, BCSide r, BCSide t, BCSide bot, BCSide f, BCSide b) 
        : left(std::move(l)), right(std::move(r)), top(std::move(t)), bottom(std::move(bot)), front(std::move(f)), back(std::move(b)){}

    // Constructor to initialize all boundaries the same way.
    BoundaryConditions3D(BCType t, BCFunc val) {
        BCSide side(t, std::move(val));

        left = side;
        right = side;
        top = side;
        bottom = side;
        front = side;
        back = std::move(side);
    }
};

// Build an RHS for the 1D heat equation using the method of lines.
// The returned RHS expects a state vector `y` containing the values at
// the `n_interior` grid points (interior points only; boundary values
// are supplied via BCs). The spatial spacing `dx` is the distance
// between adjacent nodes (interior-to-interior spacing).
//
// u_t = alpha * u_xx
//
// left_bc/right_bc select Dirichlet (prescribed u) or Neumann
// (prescribed du/dx). For Dirichlet, `left_val`/`right_val` should
// provide u(t) at the boundary. For Neumann, they should provide the
// derivative du/dx at the boundary. If a BCFunc is empty it is treated
// as zero.

RHS make_heat_rhs(double alpha,
                std::size_t n_interior,
                double dx,
                BoundaryConditions1D bc = {});


RHS make_heat_rhs(double alpha,
                std::size_t nx, std::size_t ny,
                double dx, double dy,
                BoundaryConditions2D bc = {});


RHS make_heat_rhs(double alpha,
                std::size_t nx, std::size_t ny, std::size_t nz,
                double dx, double dy, double dz,
                BoundaryConditions3D bc = {});
} // namespace pde
} // namespace diffeq
