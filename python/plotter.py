import numpy as np
import deq
print("Imported deq")
from compare_steppers import StepperComparison


"First test for Circular Motion"
def f(t, y):
    return [y[1], -y[0]]

def exact(t):
    return [np.cos(t), -np.sin(t)]

y0 = np.array([1.0, 0.0])

# "Second test for Coupled Exponential Decay"
# def f(t, y):
#     return [-y[0] + 0.5*y[1], -2*y[1]]

# def exact(t):
#     return [np.exp(-t) + 0.5 * 2 * (np.exp(-t) - np.exp(-2 * t)), 2 * np.exp(-2 * t)]

# y0 = np.array([1.0, 2.0])

"Third test for polynomial type equation"
# def f(t, y):
#     dydt = [0, 0]
#     dydt[0] = -1.0 * y[0] + t         # dy1/dt = -y1 + t
#     dydt[1] = -2.0 * y[1] + t**2     # dy2/dt = -2*y2 + t^2
#     return dydt

# def exact(t):
#     y = [0, 0]
#     y[0] = t - 1.0 + 2.0 * np.exp(-t)            # y1(t)
#     y[1] = 0.5*t**2 - 0.5*t + 0.25 - 0.25*np.exp(-2.0*t)  # y2(t)
#     return y

# y0 = np.array([1.0, 0.0])

t0 = 0.0
t_end = 20
h = 0.01
error_metric = deq.rms

comp = StepperComparison(f, exact, y0, t0, t_end, h, error_metric)

comp.add_stepper(deq.EulerStepper())
# comp.add_stepper(deq.MidpointStepper())
# comp.add_stepper(deq.RK3Stepper())
# comp.add_stepper(deq.RK4Stepper())
comp.add_stepper(deq.RK5Stepper())

comp.plot_results()