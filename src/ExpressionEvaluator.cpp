#include "ExpressionEvaluator.hpp"

#include <exprtk.hpp>

#include <stdexcept>
#include <string>
#include <map>

namespace SemiImplicitFV {

struct ExpressionEvaluator::Impl {
    using symbol_table_t = exprtk::symbol_table<double>;
    using expression_t   = exprtk::expression<double>;
    using parser_t       = exprtk::parser<double>;

    double x = 0.0, y = 0.0, z = 0.0;
    symbol_table_t symbolTable;
    std::map<std::string, expression_t> expressions;

    Impl() {
        symbolTable.add_variable("x", x);
        symbolTable.add_variable("y", y);
        symbolTable.add_variable("z", z);
        symbolTable.add_constants();  // pi, e, etc.
    }
};

ExpressionEvaluator::ExpressionEvaluator() : impl_(std::make_unique<Impl>()) {}
ExpressionEvaluator::~ExpressionEvaluator() = default;

void ExpressionEvaluator::addExpression(const std::string& name, const std::string& exprString) {
    Impl::expression_t expr;
    expr.register_symbol_table(impl_->symbolTable);

    Impl::parser_t parser;
    if (!parser.compile(exprString, expr)) {
        throw std::runtime_error("Failed to compile expression '" + name + "': "
            + parser.error() + "\n  Expression: " + exprString);
    }
    impl_->expressions[name] = std::move(expr);
}

void ExpressionEvaluator::setCoordinates(double x, double y, double z) {
    impl_->x = x;
    impl_->y = y;
    impl_->z = z;
}

double ExpressionEvaluator::evaluate(const std::string& name) const {
    auto it = impl_->expressions.find(name);
    if (it == impl_->expressions.end()) {
        throw std::runtime_error("ExpressionEvaluator: unknown expression '" + name + "'");
    }
    return it->second.value();
}

} // namespace SemiImplicitFV
