import deq
import numpy as np
import matplotlib.pyplot as plt

def run_solver(stepper, prob, t_end, h, exact, error_metric, min_pow, max_pow):
    solver = deq.Solver(stepper)
    result = solver.integrateFixedSteps(prob, t_end, h, exact, error_metric)
    convergence = solver.convergenceTest(prob, t_end, exact, min_pow, max_pow, error_metric)
    solver.printConvergenceTest(convergence)
    return result

def plot_comparison(results, exact_funcs):
    fig, axes = plt.subplots(2, len(results), figsize=(14, 8))
    fig.suptitle("Stepper Comparison", fontsize=14)

    for i, (name, (result, stepper, exact_func)) in enumerate(results.items()):
        t = np.array(result.T)
        y = np.array(result.Y)
        err = np.array(result.errors)
        y_exact = np.column_stack(exact_func(t))

        # Phase plot
        ax = axes[0, i]
        ax.plot(y[:, 0], y[:, 1], '-', label='Numerical')
        ax.plot(y_exact[:, 0], y_exact[:, 1], '--', label='Exact')
        ax.set_title(f"{stepper.GetName()}\nOrder = {stepper.GetOrder()}\nTime = {result.total_time:.4f}s")
        ax.set_xlabel("y₁")
        ax.set_ylabel("y₂")
        ax.legend()
        ax.grid(True)
        ax.axis('equal')

        # Error plot
        ax = axes[1, i]
        ax.plot(t, err)
        ax.set_title(f"{stepper.GetName()}\nError ({result.error_func_name})")
        ax.set_xlabel("Time")
        ax.set_ylabel("Error")
        ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

class StepperComparison:
        def __init__(self, f, exact, y0, t0, t_end, h, error_metric, min_pow = 1, max_pow = 7):
            self.f = f
            self.exact = exact
            self.prob = deq.IVPProblem(f, y0, t0)
            self.t_end = t_end
            self.h = h
            self.error_metric = error_metric
            self.min_pow = min_pow
            self.max_pow = max_pow
            self.results = {}

        def add_stepper(self, stepper):
             name = stepper.GetName()
             print(f"Running {name}...")
             result = run_solver(stepper, self.prob, self.t_end, self.h, self.exact, self.error_metric,
                                self.min_pow, self.max_pow)
             
             self.results[name] = (result, stepper, self.exact)
        
        def plot_results(self):
                plot_comparison(self.results, self.exact)