#ifndef EXPRESSION_BINARY_EXP_H_
#define EXPRESSION_BINARY_EXP_H_

#include "expression.h"

namespace el {

template<typename Dtype> struct Exp;
template<typename Dtype> struct BinaryExp;

namespace op {

template<typename Dtype>
struct AddExp: public BinaryExp<Dtype> {
	AddExp(const Exp<Dtype>& roperand, const Exp<Dtype>& loperand): BinaryExp<Dtype>(roperand, loperand){}
	Dtype eval(index_t* ids) const {return this->roperand_.eval(ids) + this->loperand_.eval(ids);}
};
template<typename Dtype>
inline AddExp<Dtype> operator+(const Exp<Dtype>& roperand, const Exp<Dtype>& loperand) {
	return AddExp<Dtype>(roperand, loperand);
}

template<typename Dtype>
struct SubExp: public BinaryExp<Dtype> {
	SubExp(const Exp<Dtype>& roperand, const Exp<Dtype>& loperand): BinaryExp<Dtype>(roperand, loperand){}
	Dtype eval(index_t* ids) const {return this->roperand_.eval(ids) - this->loperand_.eval(ids);}
};
template<typename Dtype>
inline SubExp<Dtype> operator-(const Exp<Dtype>& roperand, const Exp<Dtype>& loperand) {
	return SubExp<Dtype>(roperand, loperand);
}


} //namespace op

using op::operator+;
using op::operator-;
} // namespace el

#endif