#include "pde/heat_equation.hpp"
#include <cassert>

using namespace diffeq;

RHS pde::make_heat_rhs(double alpha,
                        std::size_t n_interior,
                        double dx,
                        BoundaryConditions1D bc) {
    const double coeff = alpha / (dx * dx);

    return [coeff, n_interior, dx, bc = std::move(bc)](double t, const Vec& y) -> Vec {
        assert(y.size() == n_interior);
        Vec dydt(n_interior);

        const double u_left_val = bc.left.value ? bc.left.value(t) : 0.0;
        const double u_right_val = bc.right.value ? bc.right.value(t) : 0.0;

        if (n_interior == 0) {
            return dydt;
        }

        // left boundary
        double u_im1_0;
        if (bc.left.type == pde::BCType::Dirichlet) {
            u_im1_0 = u_left_val;
        }
        else {
            const double g = u_left_val;
            u_im1_0 = (n_interior > 1) ? (y[1] - 2.0 * dx * g) : (y[0] - dx * g);
        }

        // right boundary
        double u_ip1_n;
        if (bc.right.type == pde::BCType::Dirichlet) {
            u_ip1_n = u_right_val;
        }
        else {
            const double g = u_right_val;
            u_ip1_n = (n_interior > 1) ? (y[n_interior - 2] + 2.0 * dx * g) : (y[0] + dx * g);
        }

        if (n_interior == 1) {
            dydt[0] = coeff * (u_im1_0 - 2.0 * y[0] + u_ip1_n);
            return dydt;
        }

        dydt[0] = coeff * (u_im1_0 - 2.0 * y[0] + y[1]);

        for (std::size_t i = 1; i < n_interior - 1; ++i) {
            dydt[1] = coeff * (y[i - 1] - 2.0 * y[i] + y[i + 1]);
        }

        dydt[n_interior - 1] = coeff * (y[n_interior - 2] - 2.0 * y[n_interior-1] + u_ip1_n);

        return dydt;
    };
}

RHS pde::make_heat_rhs(double alpha,
                       std::size_t nx, std::size_t ny,
                       double dx, double dy,
                       BoundaryConditions2D bc) {

    const std::size_t nx_int = nx - 2;
    const std::size_t ny_int = ny - 2;

    const double coeff_x = alpha / (dx * dx);
    const double coeff_y = alpha / (dy * dy);

    return [coeff_x, coeff_y, dx, dy, nx_int, ny_int, bc = std::move(bc)]
            (double t, const Vec& y) -> Vec {
        assert(y.size() == nx_int * ny_int);
        Vec dydt(nx_int * ny_int);

        if (nx_int == 0 || ny_int == 0) {
            return dydt;
        }

        const double bc_left = bc.left.value ? bc.left.value(t) : 0.0;
        const double bc_right = bc.right.value ? bc.right.value(t) : 0.0;
        const double bc_top = bc.top.value ? bc.top.value(t) : 0.0;
        const double bc_bottom = bc.bottom.value ? bc.bottom.value(t) : 0.0;

        auto u = [&](std::size_t i, std::size_t j) -> double {
            return y[i * ny_int + j];
        };

        auto left_ghost = [&](std::size_t j) -> double {
            if (bc.left.type == pde::BCType::Dirichlet) return bc_left;
            return (nx_int > 1) ? (u(1, j) - 2.0 * dx * bc_left) : (u(0, j) - dx * bc_left);
        };

        auto right_ghost = [&](std::size_t j) -> double {
            if (bc.right.type == pde::BCType::Dirichlet) return bc_right;
            return (nx_int > 1) ? (u(nx_int - 2, j) + 2.0 * dx * bc_right) : (u(nx_int - 1, j) + dx * bc_right);
        };

        auto top_ghost = [&](std::size_t i) -> double {
            if (bc.top.type == pde::BCType::Dirichlet) return bc_top;
            return (ny_int > 1) ? (u(i, 1) - 2.0 * dy * bc_top) : (u(i, 0) - dy * bc_top);
        };

        auto bottom_ghost = [&](std::size_t i) -> double {
            if (bc.bottom.type == pde::BCType::Dirichlet) return bc_bottom;
            return (ny_int > 1) ? (u(i, ny_int - 2) + 2.0 * dy * bc_bottom) : (u(i, ny_int - 1) + dy * bc_bottom);
        };

        // Interior block
        if (nx_int > 2 && ny_int > 2) {
            for (std::size_t i = 1; i + 1 < nx_int; ++i) { 
                for (std::size_t j = 1; j + 1 < ny_int; ++j) {
                    const double u_c = u(i, j);
                    dydt[i * ny_int + j] = 
                        coeff_x * (u(i - 1, j) - 2.0 * u_c + u(i + 1, j)) + 
                        coeff_y * (u(i, j - 1) - 2.0 * u_c + u(i, j + 1));
                }
            }
        }

        // Left and right edges (excluding corners)
        if (ny_int > 2){
            for (std::size_t j = 1; j + 1 < ny_int; ++j) {
                {   const std::size_t i = 0;
                    const double u_c = u(i, j);
                    const double u_xm = left_ghost(j);
                    const double u_xp = (nx_int > 1) ? u(1, j) : right_ghost(j);
                    dydt[i * ny_int + j] = 
                        coeff_x * (u_xm - 2.0 * u_c + u_xp) + 
                        coeff_y * (u(i, j-1) - 2.0 * u_c + u(i, j+1));
                }
                if (nx_int > 1) {
                    const std::size_t i = nx_int - 1;
                    const double u_c = u(i, j);
                    const double u_xp = right_ghost(j);
                    dydt[i * ny_int + j] = 
                        coeff_x * (u(i - 1, j) - 2.0 * u_c + u_xp) + 
                        coeff_y * (u(i, j-1) - 2.0 * u_c + u(i, j + 1));
                }
            }
        }

        // Top and bottom edges (excluding corners)
        if (nx_int > 2) {
            for (std::size_t i = 1; i + 1 < nx_int; ++i){
                {   const std::size_t j = 0;
                    const double u_c = u(i, j);
                    const double u_ym = top_ghost(i);
                    const double u_yp = (ny_int > 1) ? u(i, 1) : bottom_ghost(i);
                    dydt[i * ny_int + j] = 
                        coeff_x * (u(i-1, j) - 2.0 * u_c + u(i + 1, j)) + 
                        coeff_y * (u_ym - 2.0 * u_c + u_yp);
                }
                if (ny_int > 1) {
                    const std::size_t j = ny_int - 1;
                    const double u_c = u(i, j);
                    const double u_yp = bottom_ghost(i);
                    dydt[i * ny_int + j] =
                        coeff_x * (u(i - 1, j) - 2.0 * u_c + u(i + 1, j)) +
                        coeff_y * (u(i, j - 1) - 2.0 * u_c + u_yp);
                }
            }
        }

        // Corners
        auto corner = [&](std::size_t i, std::size_t j) {
            const double u_c = u(i, j);

            double u_xm, u_xp;
            if (i == 0) { 
                u_xm = left_ghost(i);
                u_xp = (nx_int > 1) ? u(1, j) : right_ghost(j);
            }
            else {
                u_xm = u(nx_int - 2, j);
                u_xp = right_ghost(j);
            }

            double u_ym, u_yp;
            if (j == 0) { 
                u_ym = top_ghost(i);
                u_yp = (ny_int > 1) ? u(i, 1) : bottom_ghost(i);
            }
            else {
                u_ym = u(i, ny_int - 2);
                u_yp = bottom_ghost(i);
            }

            dydt[i * ny_int + j] = 
                coeff_x * (u_xm - 2.0 * u_c + u_xp) + 
                coeff_y * (u_ym - 2.0 * u_c + u_yp);
        };

        corner(0, 0);
        if (nx_int > 1) corner(nx_int - 1, 0);
        if (ny_int > 1) corner(0, ny_int - 1);
        if (nx_int > 1 && ny_int > 1) corner(nx_int - 1, ny_int -1);

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
                        if (bc.left.type == BCType::Dirichlet) {
                            u_xm = bc_left;
                        } else if (nx_int > 1) {
                            u_xm = u(1, j, k) - 2.0 * dx * bc_left;
                        } else {
                            u_xm = u_c - dx * bc_left;
                        }
                    } else {
                        u_xm = u(i - 1, j, k);
                    }

                    double u_xp;
                    if (i == nx_int - 1) {
                        if (bc.right.type == BCType::Dirichlet) {
                            u_xp = bc_right;
                        } else if (nx_int > 1) {
                            u_xp = u(nx_int - 2, j, k) + 2.0 * dx * bc_right;
                        } else {
                            u_xp = u_c + dx * bc_right;
                        }
                    } else {
                        u_xp = u(i + 1, j, k);
                    }

                    // y-direction neighbors
                    double u_ym;
                    if (j == 0) {
                        if (bc.top.type == BCType::Dirichlet) {
                            u_ym = bc_top;
                        } else if (ny_int > 1) {
                            u_ym = u(i, 1, k) - 2.0 * dy * bc_top;
                        } else {
                            u_ym = u_c - dy * bc_top;
                        }
                    } else {
                        u_ym = u(i, j - 1, k);
                    }

                    double u_yp;
                    if (j == ny_int - 1) {
                        if (bc.bottom.type == BCType::Dirichlet) {
                            u_yp = bc_bottom;
                        } else if (ny_int > 1) {
                            u_yp = u(i, ny_int - 2, k) + 2.0 * dy * bc_bottom;
                        } else {
                            u_yp = u_c + dy * bc_bottom;
                        }
                    } else {
                        u_yp = u(i, j + 1, k);
                    }

                    // z-direction neighbors
                    double u_zm;
                    if (k == 0) {
                        if (bc.front.type == BCType::Dirichlet) {
                            u_zm = bc_front;
                        } else if (nz_int > 1) {
                            u_zm = u(i, j, 1) - 2.0 * dz * bc_front;
                        } else {
                            u_zm = u_c - dz * bc_front;
                        }
                    } else {
                        u_zm = u(i, j, k - 1);
                    }

                    double u_zp;
                    if (k == nz_int - 1) {
                        if (bc.back.type == BCType::Dirichlet) {
                            u_zp = bc_back;
                        } else if (nz_int > 1) {
                            u_zp = u(i, j, nz_int - 2) + 2.0 * dz * bc_back;
                        } else {
                            u_zp = u_c + dz * bc_back;
                        }
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
