#include "pde/heat_equation.hpp"
#include <cassert>

using namespace diffeq;

RHS pde::make_heat_rhs(double alpha,
                        std::size_t n_interior,
                        double dx,
                        BoundaryConditions1D bc) {
    return [alpha, n_interior, dx, bc = std::move(bc)](double t, const Vec& y) -> Vec {
        assert(y.size() == n_interior);
        Vec dydt(n_interior);
        const double dx2 = dx * dx;

        // Evaluate boundary prescriptions (Dirichlet values or Neumann fluxes).
        // Check std::function directly (it has an explicit operator bool).
        double u_left_val = bc.left.value ? bc.left.value(t) : 0.0;
        double u_right_val = bc.right.value ? bc.right.value(t) : 0.0;

        for (std::size_t i = 0; i < n_interior; ++i) {
            const double u_i = y[i];

            double u_im1; // u_{i-1}
            double u_ip1; // u_{i+1}

            // left neighbor
            if (i == 0) {
                if (bc.left.type == pde::BCType::Dirichlet) {
                    u_im1 = u_left_val;
                } else {
                    // Neumann: prescribe du/dx = g(t) at left boundary.
                    // Use ghost-point such that (u1 - u_ghost)/(2*dx) = g
                    // => u_ghost = u1 - 2*dx*g
                    const double g = u_left_val;
                    const double u1 = y[0];
                    const double u_ghost = u1 - 2.0 * dx * g;
                    u_im1 = u_ghost;
                }
            } else {
                u_im1 = y[i - 1];
            }

            // right neighbor
            if (i == n_interior - 1) {
                if (bc.right.type == pde::BCType::Dirichlet) {
                    u_ip1 = u_right_val;
                } else {
                    // Neumann at right: prescribe du/dx = g(t)
                    // Use ghost-point such that (u_ghost - u_{n})/(2*dx) = g
                    // => u_ghost = u_n + 2*dx*g
                    const double g = u_right_val;
                    const double u_n = y[n_interior - 1];
                    const double u_ghost = u_n + 2.0 * dx * g;
                    u_ip1 = u_ghost;
                }
            } else {
                u_ip1 = y[i + 1];
            }

            dydt[i] = alpha * (u_im1 - 2.0 * u_i + u_ip1) / dx2;
        }

        return dydt;
    };
}

RHS pde::make_heat_rhs(double alpha,
                          std::size_t nx, std::size_t ny,
                          double dx, double dy,
                          BoundaryConditions2D bc) {

    const std::size_t nx_int = nx - 2; // number of interior nodes in x
    const std::size_t ny_int = ny - 2; // number of interior nodes in y

    return [=, bc = std::move(bc)](double t, const Vec& y) -> Vec {
        assert(y.size() == nx_int * ny_int);
        Vec dydt(nx_int * ny_int);

        const double dx2 = dx * dx;
        const double dy2 = dy * dy;

        const double bc_left = bc.left.value ? bc.left.value(t) : 0.0;
        const double bc_right = bc.right.value ? bc.right.value(t) : 0.0;
        const double bc_top = bc.top.value ? bc.top.value(t) : 0.0;
        const double bc_bottom = bc.bottom.value ? bc.bottom.value(t) : 0.0;

        // Helper: get interior value by interior indices (i, j) in [0, nx_int) x [0, ny_int)
        auto u = [&](std::size_t i, std::size_t j) -> double {
            return y[i * ny_int + j];
        };

        for (std::size_t i = 0; i < nx_int; ++i) {
            for (std::size_t j = 0; j < ny_int; ++j) {
                const double u_c = u(i, j);

                // x-direction neighbors
                double u_xm; // u_{i-1, j}
                if (i == 0) {
                    if (bc.left.type == BCType::Dirichlet) {
                        u_xm = bc_left;
                    } else {
                        // Ghost point: (u(0,j) - u_ghost) / (2*dx) = g => u_ghost = u(0,j) - 2*dx*g
                        u_xm = u_c - 2.0 * dx * bc_left;
                    }
                } else {
                    u_xm = u(i - 1, j);
                }

                double u_xp; // u_{i+1, j}
                if (i == nx_int - 1) {
                    if (bc.right.type == BCType::Dirichlet) {
                        u_xp = bc_right;
                    } else {
                        // Ghost point: (u_ghost - u(nx_int-1,j)) / (2*dx) = g => u_ghost = u_c + 2*dx*g
                        u_xp = u_c + 2.0 * dx * bc_right;
                    }
                } else {
                    u_xp = u(i + 1, j);
                }

                // y-direction neighbors 
                double u_ym; // u_{i, j-1}  (top boundary, j=0)
                if (j == 0) {
                    if (bc.top.type == BCType::Dirichlet) {
                        u_ym = bc_top;
                    } else {
                        u_ym = u_c - 2.0 * dy * bc_top;
                    }
                } else {
                    u_ym = u(i, j - 1);
                }

                double u_yp; // u_{i, j+1}  (bottom boundary, j=ny_int-1)
                if (j == ny_int - 1) {
                    if (bc.bottom.type == BCType::Dirichlet) {
                        u_yp = bc_bottom;
                    } else {
                        u_yp = u_c + 2.0 * dy * bc_bottom;
                    }
                } else {
                    u_yp = u(i, j + 1);
                }

                dydt[i * ny_int + j] = alpha * (
                    (u_xm - 2.0 * u_c + u_xp) / dx2 +
                    (u_ym - 2.0 * u_c + u_yp) / dy2
                );
            }
        }

        return dydt;
    };
}

RHS pde::make_heat_rhs(double alpha,
                     std::size_t nx, std::size_t ny, std::size_t nz,
                     double dx, double dy, double dz,
                     BoundaryConditions3D bc) {

    const std::size_t nx_int = nx - 2;
    const std::size_t ny_int = ny - 2;
    const std::size_t nz_int = nz - 2;

    return [=, bc = std::move(bc)](double t, const Vec& y) -> Vec {

        assert(y.size() == nx_int * ny_int * nz_int);
        Vec dydt(nx_int * ny_int * nz_int);

        const double dx2 = dx * dx;
        const double dy2 = dy * dy;
        const double dz2 = dz * dz;

        const double bc_left = bc.left.value ? bc.left.value(t) : 0.0;
        const double bc_right = bc.right.value ? bc.right.value(t) : 0.0;
        const double bc_top = bc.top.value ? bc.top.value(t) : 0.0;
        const double bc_bottom = bc.bottom.value ? bc.bottom.value(t) : 0.0;
        const double bc_front = bc.front.value ? bc.front.value(t) : 0.0;
        const double bc_back = bc.back.value ? bc.back.value(t) : 0.0;

        // Interior index: (i, j, k) -> flat index
        // i in x, j in y, k in z
        auto u = [&](std::size_t i, std::size_t j, std::size_t k) -> double {
            return y[i * ny_int * nz_int + j * nz_int + k];
        };

        for (std::size_t i = 0; i < nx_int; ++i) {
            for (std::size_t j = 0; j < ny_int; ++j) {
                for (std::size_t k = 0; k < nz_int; ++k) {
                    const double u_c = u(i, j, k);

                    // x-direction neighbors
                    double u_xm;
                    if (i == 0) {
                        u_xm = (bc.left.type == BCType::Dirichlet)
                            ? bc_left
                            : u_c - 2.0 * dx * bc_left;
                    } else {
                        u_xm = u(i - 1, j, k);
                    }

                    double u_xp;
                    if (i == nx_int - 1) {
                        u_xp = (bc.right.type == BCType::Dirichlet)
                            ? bc_right
                            : u_c + 2.0 * dx * bc_right;
                    } else {
                        u_xp = u(i + 1, j, k);
                    }

                    // y-direction neighbors
                    double u_ym;
                    if (j == 0) {
                        u_ym = (bc.top.type == BCType::Dirichlet)
                            ? bc_top
                            : u_c - 2.0 * dy * bc_top;
                    } else {
                        u_ym = u(i, j - 1, k);
                    }

                    double u_yp;
                    if (j == ny_int - 1) {
                        u_yp = (bc.bottom.type == BCType::Dirichlet)
                            ? bc_bottom
                            : u_c + 2.0 * dy * bc_bottom;
                    } else {
                        u_yp = u(i, j + 1, k);
                    }

                    // z-direction neighbors
                    double u_zm;
                    if (k == 0) {
                        u_zm = (bc.front.type == BCType::Dirichlet)
                            ? bc_front
                            : u_c - 2.0 * dz * bc_front;
                    } else {
                        u_zm = u(i, j, k - 1);
                    }

                    double u_zp;
                    if (k == nz_int - 1) {
                        u_zp = (bc.back.type == BCType::Dirichlet)
                            ? bc_back
                            : u_c + 2.0 * dz * bc_back;
                    } else {
                        u_zp = u(i, j, k + 1);
                    }

                    dydt[i * ny_int * nz_int + j * nz_int + k] = alpha * (
                        (u_xm - 2.0 * u_c + u_xp) / dx2 +
                        (u_ym - 2.0 * u_c + u_yp) / dy2 +
                        (u_zm - 2.0 * u_c + u_zp) / dz2
                    );
                }
            }
        }

        return dydt;
    };
}
