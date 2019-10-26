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
	CHECK_OPERATOR_BROADCAST(roperand, loperand);
	return AddExp<Dtype>(roperand, loperand);
}

template<typename Dtype>
struct SubExp: public BinaryExp<Dtype> {
	SubExp(const Exp<Dtype>& roperand, const Exp<Dtype>& loperand): BinaryExp<Dtype>(roperand, loperand){}
	Dtype eval(index_t* ids) const {return this->roperand_.eval(ids) - this->loperand_.eval(ids);}
};
template<typename Dtype>
inline SubExp<Dtype> operator-(const Exp<Dtype>& roperand, const Exp<Dtype>& loperand) {
	CHECK_OPERATOR_BROADCAST(roperand, loperand);
	return SubExp<Dtype>(roperand, loperand);
}

template<typename Dtype>
struct MMExp: public BinaryExp<Dtype> {
	MMExp(const Exp<Dtype>& roperand, const Exp<Dtype>& loperand): BinaryExp<Dtype>(roperand, loperand){}
	index_t dim(void) const {return 2;}
	index_t size(index_t idx) const {return idx == 0 ? this->roperand_.size(0) : this->loperand_.size(1);}
	Dtype eval(index_t* ids) const {
		Dtype value = 0;
		index_t r_loc[2] = {ids[0], 0};
		index_t l_loc[2] = {0, ids[1]};
		for(index_t i = 0; i < this->roperand_.size(1); i++) {
			r_loc[1] = i;
			l_loc[0] = i;
			value += this->roperand_.eval(r_loc) * this->loperand_.eval(l_loc);
		}
		return value;
	}
};
template<typename Dtype>
inline MMExp<Dtype> mm(const Exp<Dtype>& roperand, const Exp<Dtype>& loperand) {
	CHECK_DIM_MATCH(roperand.dim(), 2);
	CHECK_DIM_MATCH(loperand.dim(), 2);
	CHECK_SIZE_EQUAL(roperand.size(1), loperand.size(0));
	return MMExp<Dtype>(roperand, loperand);
}

} //namespace op

using op::operator+;
using op::operator-;
} // namespace el

#endif