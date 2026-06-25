import numpy as np
import deq
import matplotlib.pyplot as plt

c, k = 0.2, 1.0

def f(t, y):
    dydt = [0, 0]
    dydt[0] = y[1]
    dydt[1] = -k * y[0] - c * y[1]
    return dydt

y0 = [1.0, 0.0]
t0 = 0.0
t_end = 10
h = 0.01

prob = deq.IVPProblem(f, y0, t0)

rk4_stepper = deq.RK4Stepper()
solver = deq.Solver(rk4_stepper)

result = solver.integrateFixedSteps(prob, t_end, h)

y_vals, dydt_vals = np.transpose(result.Y)

plt.plot(result.T, y_vals, label='Position')
plt.plot(result.T, dydt_vals, label="Speed")
plt.grid(True)
plt.legend()
plt.show()