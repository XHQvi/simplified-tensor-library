#ifndef EXPRESSION_UNARY_EXP_H_
#define EXPRESSION_UNARY_EXP_H_

#include <cmath>
#include "expression.h"

namespace el {

template<typename Dtype> struct Exp;
template<typename Dtype> struct UnaryExp;

namespace op {

template<typename Dtype>
struct AbsExp: public UnaryExp<Dtype> {
	explicit AbsExp(const Exp<Dtype>& operand): UnaryExp<Dtype>(operand) {}
	Dtype eval(index_t* ids) const {return std::abs(this->operand_.eval(ids));}
};
template<typename Dtype>
inline AbsExp<Dtype> abs(const Exp<Dtype>& operand) {
	return AbsExp<Dtype>(operand);
}

template<typename Dtype>
struct SigmoidExp: public UnaryExp<Dtype> {
	explicit SigmoidExp(const Exp<Dtype>& operand): UnaryExp<Dtype>(operand) {}
	Dtype eval(index_t* ids) const {return 1 / (1+std::exp(-this->operand_.eval(ids)));}
};
template<typename Dtype>
inline SigmoidExp<Dtype> sigmoid(const Exp<Dtype>& operand) {
	return SigmoidExp<Dtype>(operand);
}

} // namespace op
} // namespace el

#endif