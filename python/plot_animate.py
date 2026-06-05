import diffeqpy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec

# Define ODE
def f(t, y):
    return [y[1], -y[0]]

def exact(t):
    return [np.cos(t), -np.sin(t)]

y0 = np.array([1.0, 0.0])
t0 = 0.0
t_end = 10.0
h = 0.01

# Solve numerically
prob = diffeqpy.IVPProblem(f, y0, t0)
stepper = diffeqpy.RK5Stepper()
solver = diffeqpy.Solver(stepper)
result = solver.integrateFixedSteps(prob, t_end, h, exact)

t_vals = np.array(result.T)
y_vals = np.array(result.Y)
errors = np.array(result.errors)  # numerical error for each step

# Exact solution
y_exact = np.column_stack([np.cos(t_vals), -np.sin(t_vals)])

# Figure with gridspec: top plot bigger
fig = plt.figure(figsize=(7, 8))
gs = GridSpec(3, 1, figure=fig, hspace=0.4)
ax1 = fig.add_subplot(gs[:2, 0])  # top 2/3 for trajectory
ax2 = fig.add_subplot(gs[2, 0])  # bottom 1/3 for error

# Top plot: trajectory
ax1.set_xlim(-1.2, 1.2)
ax1.set_ylim(-1.2, 1.2)
ax1.set_xlabel('y1')
ax1.set_ylabel('y2')
ax1.set_title('Circular motion in 2D')
ax1.grid(True)
ax1.set_aspect('equal')

line_num, = ax1.plot([], [], 'r-', lw=2, label='Numerical trajectory')
line_exact, = ax1.plot([], [], 'b--', lw=1, label='Exact trajectory')
point_num, = ax1.plot([], [], 'ro', label='Numerical position')
point_exact, = ax1.plot([], [], 'bo', label='Exact position')
ax1.legend()

# Bottom plot: error vs time
ax2.set_xlim(t_vals[0], t_vals[-1])
ax2.set_ylim(0, np.max(errors)*1.1)
ax2.set_xlabel('Time')
ax2.set_ylabel('Error')
ax2.set_title(f'{result.error_func_name} vs Time')
ax2.grid(True)
line_error, = ax2.plot([], [], 'm-', lw=2)

# Animation function
def animate(i):
    # Top plot
    line_num.set_data(y_vals[:i, 0], y_vals[:i, 1])
    line_exact.set_data(y_exact[:i, 0], y_exact[:i, 1])
    point_num.set_data([y_vals[i, 0]], [y_vals[i, 1]])
    point_exact.set_data([y_exact[i, 0]], [y_exact[i, 1]])
    # Bottom plot
    line_error.set_data(t_vals[:i], errors[:i])
    return line_num, line_exact, point_num, point_exact, line_error

ani = FuncAnimation(fig, animate, frames=len(t_vals), interval=10, blit=True)
plt.show()
