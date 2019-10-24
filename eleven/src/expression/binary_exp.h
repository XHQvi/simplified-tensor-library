#ifndef EXPRESSION_BINARY_EXP_H_
#define EXPRESSION_BINARY_EXP_H_

#include "expression.h"

namespace el {

template<typename SubType, typename Dtype> struct Exp;
template<typename OP, typename ROtype, typename LOtype, typename Dtype> struct BinaryMapExp;

namespace op {

template<typename Dtype>
struct AddOP {
	static Dtype map(Dtype rvalue, Dtype lvalue) {return rvalue + lvalue;}
};
template<typename ROtype, typename LOtype, typename Dtype>
inline BinaryMapExp<AddOP<Dtype>, ROtype, LOtype, Dtype> operator+(const Exp<ROtype, Dtype>& roperand, const Exp<LOtype, Dtype>& loperand) {
	return BinaryMapExp<AddOP<Dtype>, ROtype, LOtype, Dtype>(roperand.self(), loperand.self());
}

template<typename Dtype>
struct SubOP {
	static Dtype map(Dtype rvalue, Dtype lvalue) {return rvalue - lvalue;}
};
template<typename ROtype, typename LOtype, typename Dtype>
inline BinaryMapExp<SubOP<Dtype>, ROtype, LOtype, Dtype> operator-(const Exp<ROtype, Dtype>& roperand, const Exp<LOtype, Dtype>& loperand) {
	return BinaryMapExp<SubOP<Dtype>, ROtype, LOtype, Dtype>(roperand.self(), loperand.self());
}


} //namespace op

using op::operator+;
using op::operator-;
} // namespace el

#endif