#ifndef EXPRESSION_UNARY_EXP_H_
#define EXPRESSION_UNARY_EXP_H_

#include <cmath>
#include "expression.h"

namespace el {

template<typename SubType, typename Dtype> struct Exp;
template<typename OP, typename Otype, typename Dtype> struct UnaryMapExp;

namespace op {

template<typename Dtype>
struct AbsOP {
	static Dtype map(Dtype value) {return value > 0 ? value: (-value);}
};
template<typename Otype, typename Dtype>
inline UnaryMapExp<AbsOP<Dtype>, Otype, Dtype> abs(const Exp<Otype, Dtype>& operand) {
	return UnaryMapExp<AbsOP<Dtype>, Otype, Dtype>(operand.self());
}

template<typename Dtype>
struct SigmoidOP {
	static Dtype map(Dtype value) {return 1 / (1+std::exp(-value));}
};
template<typename Otype, typename Dtype>
inline UnaryMapExp<SigmoidOP<Dtype>, Otype, Dtype> sigmoid(const Exp<Otype, Dtype>& operand) {
	return UnaryMapExp<SigmoidOP<Dtype>, Otype, Dtype>(operand.self());
}

} // namespace op
} // namespace el

#endif