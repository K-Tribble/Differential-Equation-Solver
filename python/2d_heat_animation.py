import deq
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.animation import FuncAnimation

"""
Code for plotting temperature diffusion
on a 2D rectangular plate over time.
Boundary conditions : Constant 293K temperature at the edges
Initial Profile : Gaussian Initial Profile
"""

alpha = 0.5

# Define the rectangle grid
N = 100
dx, dy = 1 / N, 1 / N
x = np.linspace(-1/2, 1/2, N)
y = np.linspace(-1/2, 1/2, N)

X, Y = np.meshgrid(x, y)

# Gaussian Initial Profile 
sigma = 0.3
u0_full = 293 - 40 * np.exp(-(X**2 + Y**2) / sigma**2)

print(f"Initial Mean (full grid) = {np.mean(u0_full)}")

# Extract interior only
u0_int = u0_full[1:-1, 1:-1]

# Flatten for solver
y0 = u0_int.flatten()

def bc_func(x):
    return 293

t0 = 0

bc = deq.BoundaryConditions2D(deq.BCType.Dirichlet, bc_func)

rhs = deq.make_heat_rhs(alpha, N, N, dx, dy, bc)

prob = deq.IVPProblem(rhs, y0, 0)

stepper = deq.RK5Stepper()
solver = deq.Solver(stepper)

# Stability limit is given y dt <= dx^2/(4 * alpha), this is under the limit
# General stability limit is dt <= dx^2(2 * alpha * D), D=number of dimensions
def stability_limit2d(dx, alpha):
    return dx * dx / (4 * alpha)

stab_lim = stability_limit2d(dx,  alpha)
print(f"stability limit = {stab_lim}")
dt = 0.4 * stab_lim
print(f"dt = {dt}")

t_end = 0.5

result = solver.integrateFixedSteps(prob, t_end, dt, deq.history_level.full)

result.print_info()

# Time points
t_points = np.array(result.T)
# Numerical Solutions (list of arrays at for solutions at each time point)
u_vecs = np.array(result.Y) 
# Reshape soltuions to grid
u_vecs = np.array([u.reshape((N-2, N-2)) for u in u_vecs])

# Mean temperature of the distribution, should converge to 293K
u_means = [np.mean(u_vec) for u_vec in u_vecs]

# Reconstruct full grid with Neumann BC
u_plots = np.array([np.zeros((N, N)) for _ in u_vecs])
for u_vec, u_plot in zip(u_vecs, u_plots):
    u_plot[1:-1, 1:-1] = u_vec

    # For Neumann BC, copy boundary from nearest interior
    u_plot[0, :] = 293
    u_plot[-1, :] = 293
    u_plot[:, 0] = 293
    u_plot[:, -1] = 293


# Plots: a horizontal subplot, the left plot is an animation of the temperature over time
# the right plot is the mean temperature plotted over timey
fig, (ax_heat, ax_mean) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Heat Diffusion on a 2D Plate — Dirichlet BC', fontsize=13)

# Animated heatmap
vmin, vmax = u_plots.min(), u_plots.max()

im = ax_heat.imshow(
    u_plots[0],
    extent=[-0.5, 0.5, -0.5, 0.5],
    origin='lower',
    cmap='inferno',
    vmin=vmin,
    vmax=vmax,
    aspect='equal',
)
cbar = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
cbar.set_label('Temperature (K)')
ax_heat.set_xlabel('x (m)')
ax_heat.set_ylabel('y (m)')
time_label = ax_heat.set_title(f't = {t_points[0]:.5f} s')

# Mean temperature vs time
ax_mean.plot(t_points, u_means, color='steelblue', lw=1.5)
ax_mean.set_xlabel('Time (s)')
ax_mean.set_ylabel('Mean Temperature (K)')
ax_mean.set_title('Domain-Averaged Temperature vs Time')
ax_mean.set_xlim(t_points[0], t_points[-1])

# Small y-buffer so a near-flat conservation line is still visible
y_pad = max((max(u_means) - min(u_means)) * 0.1, 1e-6)
ax_mean.set_ylim(min(u_means) - y_pad, max(u_means) + y_pad)
ax_mean.yaxis.set_major_formatter(FormatStrFormatter('%.6f'))

# Moving time indicator
time_dot,  = ax_mean.plot(t_points[0], u_means[0], 'o', color='tomato', ms=7, zorder=5)
time_vline = ax_mean.axvline(t_points[0], color='tomato', ls='--', lw=1, alpha=0.7)

plt.tight_layout()

n_total = len(t_points)
step = max(1, n_total // 300)
frame_indices = list(range(0, n_total, step))

def update(frame):
    im.set_data(u_plots[frame])
    time_label.set_text(f't = {t_points[frame]:.5f} s')
    time_dot.set_data([t_points[frame]], [u_means[frame]])
    time_vline.set_xdata([t_points[frame]])
    return im, time_label, time_dot, time_vline

anim = FuncAnimation(
    fig, update,
    frames=frame_indices,
    interval=40,  
    blit=True,
)

anim.save('Plots/Animations/heat_diffusion.gif', writer='pillow', fps=25)
plt.show()