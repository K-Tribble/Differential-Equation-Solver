import deq
import numpy as np
import matplotlib.pyplot as plt

"""
Code for plotting temperature diffusion
on a 2D rectangular plate over time.
Boundary conditions : No heat flow at the edges
Initial Profile : Gaussian Initial Profile
"""

alpha = 1

# Define the rectangle grid
N = 10
dx, dy = 1 / N, 1 / N
x = np.linspace(-1/2, 1/2, N)
y = np.linspace(-1/2, 1/2, N)

X, Y = np.meshgrid(x, y)

# Gaussian Initial Profile 
sigma = 0.25
u0_full = np.exp(-(X**2 + Y**2) / sigma**2)

print("Initial Mean (full grid) =", np.mean(u0_full))

# Extract interior only
u0_int = u0_full[1:-1, 1:-1]

# Flatten for solver
y0 = u0_int.flatten()

def bc_func(x):
    return 0

t0 = 0

bc = deq.BoundaryConditions2D(deq.BCType.Neumann, bc_func)

rhs = deq.make_heat_rhs(alpha, N, N, dx, dy, bc)

prob = deq.IVPProblem(rhs, y0, 0)

stepper = deq.RK3Stepper()
solver = deq.Solver(stepper)

# Stability limit is given y dt <= dx^2/(4 * alpha), this is under the limit
dt = 1.5e-5

t_end = 5
nsteps = int(t_end / dt)

result = solver.integrateFixedSteps(prob, t_end, dt, deq.history_level.final_only)

result.print_info()

# Extract final state
u_final = np.array(result.Y)[-1]
u_final = u_final.reshape((N-2, N-2))
print(f"Final mean = {np.mean(u_final)}")
print(f"Time take = {result.total_time}")

# Reconstruct full grid (with Neumann BC)
u_plot = np.zeros((N, N))
u_plot[1:-1, 1:-1] = u_final

# Neumann: copy boundary from nearest interior
u_plot[0, :] = u_plot[1, :]
u_plot[-1, :] = u_plot[-2, :]
u_plot[:, 0] = u_plot[:, 1] 
u_plot[:, -1] = u_plot[:, -2]

plt.figure(figsize=(6,5))
plt.imshow(u_plot, extent=[-0.5,0.5,-0.5,0.5], origin='lower')
plt.colorbar(label="Temperature")
plt.title(f"Heat Diffusion at t = {t_end}")
plt.xlabel("x")
plt.ylabel("y")
plt.show()