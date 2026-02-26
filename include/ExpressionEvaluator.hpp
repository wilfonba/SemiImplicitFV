#ifndef EXPRESSION_EVALUATOR_HPP
#define EXPRESSION_EVALUATOR_HPP

#include <memory>
#include <string>
#include <map>

/// Evaluates math expressions with variables x, y, z.
/// Uses pimpl pattern to isolate the exprtk header (40K+ lines) to one TU.
class ExpressionEvaluator {
public:
    ExpressionEvaluator();
    ~ExpressionEvaluator();

    // Non-copyable (owns compiled expressions)
    ExpressionEvaluator(const ExpressionEvaluator&) = delete;
    ExpressionEvaluator& operator=(const ExpressionEvaluator&) = delete;

    /// Add a named expression (e.g., "rho", "1.0 + 0.5*sin(2*pi*x)").
    /// Must be called before evaluate().
    void addExpression(const std::string& name, const std::string& exprString);

    /// Set the coordinate variables for the next evaluation.
    void setCoordinates(double x, double y, double z);

    /// Evaluate a previously-added named expression. Returns the result.
    double evaluate(const std::string& name) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

#endif // EXPRESSION_EVALUATOR_HPP
