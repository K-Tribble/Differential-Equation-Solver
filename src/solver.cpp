#include "solver.hpp"
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <limits>
#include <chrono>

using namespace diffeq;

// Integrate with exact solution
IntegrationResult Solver::integrateFixedSteps(const IVPProblem& prob, double t_end, double h, const std::function<Vec(double)>& exactSolution, const errorfunc::ErrorMetric& metric, HistoryLevel history) const {
    auto startTime = std::chrono::steady_clock::now();
    IntegrationResult result;
    result.h_used = h;
    result.error_func_name = metric.name;

    double t = prob.t0;
    Vec y = prob.y0;

    result.T.push_back(t);
    result.Y.push_back(y);
    result.exactY.push_back(exactSolution(t));
    result.errors.push_back(metric.func(y, result.exactY.back()));

    const double epsilon = std::numeric_limits<double>::epsilon() * std::max<double>(1.0, std::abs(t_end));

    while (t < t_end - epsilon) {
        double hstep = std::min(h, t_end - t);
        stepper.step(prob.f, t, y, hstep);
        
        if (history == HistoryLevel::FULL) { // Store every step in Y
            result.T.push_back(t);
            result.Y.push_back(y);
        } else { // For HistoryLevel::FINAL_ONLY
            // Only store final step
            if (t >= t_end - epsilon) {
                result.T.push_back(t);
                result.Y.push_back(y);
            }
        }


        Vec exact = exactSolution(t);
        result.exactY.push_back(exact);
        result.errors.push_back(metric.func(y, exact));

        result.n_steps++;
    }

    result.final_error = result.errors.back();

    auto endTime = std::chrono::steady_clock::now();
    result.total_time = std::chrono::duration<double>(endTime - startTime).count();

    result.exact_func_given = true;

    return result;
}

// Integrate without exact solution
IntegrationResult Solver::integrateFixedSteps(const IVPProblem& prob, double t_end, double h, HistoryLevel history) const {
    auto startTime = std::chrono::steady_clock::now();
    IntegrationResult result;
    result.h_used = h;
    result.error_func_name = "None";

    double t = prob.t0;
    Vec y = prob.y0;

    result.T.push_back(t);
    result.Y.push_back(y);

    const double epsilon = std::numeric_limits<double>::epsilon() * std::max<double>(1.0, std::abs(t_end));

    while (t < t_end - epsilon) {
        double hstep = std::min(h, t_end - t);
        stepper.step(prob.f, t, y, hstep);

        if (history == HistoryLevel::FULL) { // Store every step in Y
            result.T.push_back(t);
            result.Y.push_back(y);
        } else { // For HistoryLevel::FINAL_ONLY
            // Only store final step
            if (t >= t_end - epsilon) {
                result.T.push_back(t);
                result.Y.push_back(y);
            }
        }
        result.n_steps++;
    }

    auto endTime = std::chrono::steady_clock::now();
    result.total_time = std::chrono::duration<double>(endTime - startTime).count();

    result.exact_func_given = false;

    return result;
}

void Solver::printResults(const IntegrationResult& result) const {
    bool hasExact = !result.exactY.empty() && !result.errors.empty();

    std::cout << "\n--- Integration Results ---\n";
    std::cout << "Step size h = " << result.h_used << "\n";
    std::cout << "Total steps: " << result.n_steps << "\n";
    std::cout << "Total runtime: " << result.total_time << " s\n";

    if (hasExact)
        std::cout << "Error Metric: " << result.error_func_name << "\n\n";

    std::cout << "t\t\ty_num";
    if (hasExact) std::cout << "\t\ty_exact\t\terror";
    std::cout << "\n";

    for (size_t i = 0; i < result.T.size(); ++i) {
        std::cout << std::fixed << std::setprecision(6) << result.T[i] << "\t[";

        for (size_t j = 0; j < result.Y[i].size(); ++j)
            std::cout << result.Y[i][j] << (j + 1 < result.Y[i].size() ? ", " : "");

        std::cout << "]";
        if (hasExact) {
            std::cout << "  [";
            for (size_t j = 0; j < result.exactY[i].size(); ++j)
                std::cout << result.exactY[i][j] << (j + 1 < result.exactY[i].size() ? ", " : "");
            std::cout << "]  " << result.errors[i];
        }
        std::cout << "\n";
    }
}

void Solver::demoRun(const IVPProblem& prob, double t_end, double h, const std::function<Vec(double)>& exactSolution, const errorfunc::ErrorMetric& metric) const {
    auto result = integrateFixedSteps(prob, t_end, h, exactSolution, metric);
    printResults(result);
}

ConvergenceTestResult Solver::convergenceTest( const IVPProblem& prob, double t_end, const std::function<Vec(double)>& exactSolution, int minPow, int maxPow, const errorfunc::ErrorMetric& metric) const {
    ConvergenceTestResult result;
    result.t_end = t_end;
    result.error_name = metric.name;
    result.min_pow = minPow;
    result.max_pow = maxPow;

    double prev_err = -1.0;

    for (int k = minPow; k <= maxPow; ++k) {
        double h = std::pow(0.5, k);
        result.h_vals.push_back(h);

        auto integration = integrateFixedSteps(prob, t_end, h, exactSolution, metric);
        double err = integration.final_error;
        result.final_error.push_back(err);

        if (prev_err > 0.0) {
            double ratio = prev_err / err;
            double p = std::log(ratio) / std::log(2.0);
            result.final_error_ratios.push_back(ratio);
            result.p_estimations.push_back(p);
        } else {
            result.final_error_ratios.push_back(-1.0);
            result.p_estimations.push_back(0.0);
        }
        prev_err = err;
    }

    return result;
}

void Solver::printConvergenceTest(const ConvergenceTestResult& result) const {
    std::cout << "\n--- Convergence Test ---\n";
    std::cout << "Method: " << stepper.GetName() << "\n";
    std::cout << "Order:  " << stepper.GetOrder() << "\n";
    std::cout << "Metric: " << result.error_name << "\n";
    std::cout << "t_end = " << result.t_end << "\n\n";

    std::cout << std::setw(12) << "h"
              << std::setw(20) << "final error"
              << std::setw(20) << "ratio"
              << std::setw(20) << "p_est\n";

    for (size_t i = 0; i < result.h_vals.size(); ++i) {
        std::cout << std::scientific << std::setprecision(3)
                  << std::setw(12) << result.h_vals[i]
                  << std::setw(20) << result.final_error[i];

        if (result.final_error_ratios[i] > 0.0)
            std::cout << std::setw(20) << result.final_error_ratios[i]
                      << std::setw(20) << result.p_estimations[i];
        else
            std::cout << std::setw(20) << "nan"
                      << std::setw(20) << "nan";

        std::cout << "\n";
    }
}

void Solver::convergenceTestDemoRun( const IVPProblem& prob, double t_end, const std::function<Vec(double)>& exactSolution, int minPow, int maxPow, const errorfunc::ErrorMetric& metric) const {
    auto result = convergenceTest(prob, t_end, exactSolution, minPow, maxPow, metric);
    printConvergenceTest(result);
}

std::vector<Vec> Solver::GetApproximations(const IVPProblem& prob, double t_end, double h) const {
    return integrateFixedSteps(prob, t_end, h).Y;
}

std::vector<Vec> Solver::GetExactVals(const std::vector<double>& T, const std::function<Vec(double)>& exactSolution) const {
    std::vector<Vec> exact;
    exact.reserve(T.size());
    for (double t : T)
        exact.push_back(exactSolution(t));
    return exact;
}

std::vector<double> Solver::GetErrors(const std::vector<Vec>& Y_num, const std::vector<Vec>& Y_exact, const errorfunc::ErrorMetric& metric) const {
    std::vector<double> errs;
    errs.reserve(Y_num.size());
    for (size_t i = 0; i < Y_num.size(); ++i)
        errs.push_back(metric.func(Y_num[i], Y_exact[i]));
    return errs;
}
