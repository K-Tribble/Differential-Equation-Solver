import numpy as np
import deq
import matplotlib.pyplot as plt

# Predator-prey system (no exact solution)
def f(t, y):
    dydt = [0, 0]
    dydt[0] = y[0] * (1 - y[1])    # Prey equation
    dydt[1] = y[1] * (y[0] - 1)    # Predator equation
    return dydt

y0 = [1.2, 0.8]
t0 = 0
t_end = 20
h = 0.01

prob = deq.IVPProblem(f, y0, t0)

euler_stepper = deq.EulerStepper()
euler_solver = deq.Solver(euler_stepper)
result_euler = euler_solver.integrateFixedSteps(prob, t_end, h)

rk5_stepper = deq.RK5Stepper()
rk5_solver = deq.Solver(rk5_stepper)
result_rk5 = rk5_solver.integrateFixedSteps(prob, t_end, h)

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].plot(result_euler.T_np, result_euler.Y_np[:, 0], label='y1 (prey, Euler)', color='C0')
axs[0].plot(result_euler.T_np, result_euler.Y_np[:, 1], label='y2 (predator, Euler)', color='C1')
axs[0].set_title('Euler Method')
axs[0].set_xlabel('t')
axs[0].set_ylabel('Population')
axs[0].legend()
axs[0].grid(True)

axs[1].plot(result_euler.Y_np[:, 0], result_euler.Y_np[:, 1], label='Euler', color='C2', alpha=0.7)
axs[1].plot(result_rk5.Y_np[:, 0], result_rk5.Y_np[:, 1], label='RK5', color='C3')
axs[1].set_title('Phase Portrait (Predator–Prey)')
axs[1].set_xlabel('Prey population (y1)')
axs[1].set_ylabel('Predator population (y2)')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()
