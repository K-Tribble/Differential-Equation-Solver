#pragma once
#include "problem.hpp"
#include "Steppers/stepper.hpp"
#include <vector>
#include "errorfunc/funcs.hpp"
#include "errorfunc/metric.hpp"

namespace diffeq {

enum class HistoryLevel {
    FINAL_ONLY, 
    FULL // Every Step
};

struct IntegrationResult {
    std::vector<double> T;              // Time points
    std::vector<Vec> Y;                 // Numerical solutions for all T
    std::vector<Vec> exactY;            // Exact solutions (if provided)
    std::vector<double> errors;         // Error values per step
    std::string error_func_name;        // Name of error metric
    int n_steps = 0;                    // Number of steps
    double h_used = 0.0;                // Step size used
    double total_time = 0.0;            // Execution time (seconds)
    double final_error = 0.0;            // Error at final time
    HistoryLevel history_level;         // How much of the history was stored
};

struct ConvergenceTestResult {
    double t_end = 0.0;
    std::vector<double> h_vals;             // Step sizes
    std::vector<double> final_error;         // Error for each h
    std::vector<double> final_error_ratios;  // Ratio of successive errors
    std::vector<double> p_estimations;      // Estimated order
    std::string error_name;                 // Metric name
    int min_pow = 0;
    int max_pow = 0;
};

class Solver {
public:
    explicit Solver(Stepper& stepper_) : stepper(stepper_) {}

    // Integrate with exact solution (collects errors)
    IntegrationResult integrateFixedSteps(const IVPProblem& prob, double t_end, double h, const std::function<Vec(double)>& exactSolution, const errorfunc::ErrorMetric& metric = errorfunc::L2, HistoryLevel history = HistoryLevel::FULL) const;

    // Integrate without exact solution (no errors)
    IntegrationResult integrateFixedSteps(const IVPProblem& prob, double t_end, double h, HistoryLevel history = HistoryLevel::FULL) const;

    // Print results (with or without exact solution)
    void printResults(const IntegrationResult& result) const;

    // Demo integration run
    void demoRun(const IVPProblem& prob, double t_end, double h, const std::function<Vec(double)>& exactSolution, const errorfunc::ErrorMetric& metric = errorfunc::L2) const;

    // Convergence test (varying step sizes)
    ConvergenceTestResult convergenceTest(const IVPProblem& prob, double t_end, const std::function<Vec(double)>& exactSolution, int minPow = 1, int maxPow = 7, const errorfunc::ErrorMetric& metric = errorfunc::L2) const;

    // Print convergence results
    void printConvergenceTest(const ConvergenceTestResult& result) const;

    // Demo convergence test
    void convergenceTestDemoRun(const IVPProblem& prob, double t_end, const std::function<Vec(double)>& exactSolution, int minPow = 1, int maxPow = 7, const errorfunc::ErrorMetric& metric = errorfunc::L2) const;

    // Utility functions
    std::vector<Vec> GetApproximations(const IVPProblem& prob, double t_end, double h) const;
    std::vector<Vec> GetExactVals(const std::vector<double>& T, const std::function<Vec(double)>& exactSolution) const;
    std::vector<double> GetErrors(const std::vector<Vec>& Y_num, const std::vector<Vec>& Y_exact, const errorfunc::ErrorMetric& metric = errorfunc::L2) const;

private:
    Stepper& stepper;
};

}
