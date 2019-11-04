#ifndef EXPRESSION_BINARY_EXP_H_
#define EXPRESSION_BINARY_EXP_H_

#include "expression.h"

namespace el {

template<typename Dtype> struct Exp;
template<typename Dtype> struct BinaryExp;

namespace op {

template<typename Dtype>
struct AddExp: public BinaryExp<Dtype> {
	AddExp(const Exp<Dtype>& loperand, const Exp<Dtype>& roperand): BinaryExp<Dtype>(loperand, roperand){}
	Dtype eval(index_t* ids) const {return this->loperand_.eval(ids) + this->roperand_.eval(ids);}
};
template<typename Dtype>
inline AddExp<Dtype> operator+(const Exp<Dtype>& loperand, const Exp<Dtype>& roperand) {
	CHECK_BROADCAST(loperand, roperand);
	return AddExp<Dtype>(loperand, roperand);
}

template<typename Dtype>
struct SubExp: public BinaryExp<Dtype> {
	SubExp(const Exp<Dtype>& loperand, const Exp<Dtype>& roperand): BinaryExp<Dtype>(loperand, roperand){}
	Dtype eval(index_t* ids) const {return this->loperand_.eval(ids) - this->roperand_.eval(ids);}
};
template<typename Dtype>
inline SubExp<Dtype> operator-(const Exp<Dtype>& loperand, const Exp<Dtype>& roperand) {
	CHECK_BROADCAST(loperand, roperand);
	return SubExp<Dtype>(loperand, roperand);
}

template<typename Dtype>
struct MMExp: public BinaryExp<Dtype> {
	MMExp(const Exp<Dtype>& loperand, const Exp<Dtype>& roperand): BinaryExp<Dtype>(loperand, roperand){}
	index_t dim(void) const {return 2;}
	index_t size(index_t idx) const {return idx == 0 ? this->loperand_.size(0) : this->roperand_.size(1);}
	Dtype eval(index_t* ids) const {
		Dtype value = 0;
		index_t l_loc[2] = {ids[0], 0};
		index_t r_loc[2] = {0, ids[1]};
		for(index_t i = 0; i < this->loperand_.size(1); i++) {
			l_loc[1] = i;
			r_loc[0] = i;
			value += this->loperand_.eval(l_loc) * this->roperand_.eval(r_loc);
		}
		return value;
	}
};
template<typename Dtype>
inline MMExp<Dtype> mm(const Exp<Dtype>& loperand, const Exp<Dtype>& roperand) {
	CHECK_EQUAL(loperand.dim(), 2, OperandSizeNotMatch,
		"MM need 2D Tensor, but got %dD.", loperand.dim());
	CHECK_EQUAL(roperand.dim(), 2, OperandSizeNotMatch,
		"MM need 2D Tensor, but got %dD.", roperand.dim());
	CHECK_EQUAL(loperand.size(1), roperand.size(0), OperandSizeNotMatch,
		"MM need lsize(1) and rsize(0) equal, but got size %d and %d.", loperand.size(1), roperand.size(0));
	return MMExp<Dtype>(loperand, roperand);
}

template<typename Dtype>
struct BMMExp: public BinaryExp<Dtype> {
	BMMExp(const Exp<Dtype>& loperand, const Exp<Dtype>& roperand): BinaryExp<Dtype>(loperand, roperand){}
	index_t dim(void) const {return 3;}
	index_t size(index_t idx) const {
		switch(idx) {
			case 0: return std::max(this->roperand_.size(0), this->loperand_.size(0));
			case 1: return this->loperand_.size(1);
			default: return this->roperand_.size(2);
		}
	}
	Dtype eval(index_t* ids) const {
		Dtype value = 0;
		index_t l_loc[3] = {ids[0], ids[1], 0};
		index_t r_loc[3] = {ids[0], 0, ids[2]};
		for(index_t i = 0; i < this->loperand_.size(2); i++) {
			l_loc[2] = i;
			r_loc[1] = i;
			value += this->loperand_.eval(l_loc) * this->roperand_.eval(r_loc);
		}
		return value;
	}
};
template<typename Dtype>
inline BMMExp<Dtype> bmm(const Exp<Dtype>& loperand, const Exp<Dtype>& roperand) {
	CHECK_EQUAL(loperand.dim(), 3, OperandSizeNotMatch,
		"BMM need 2D Tensor, but got %dD.", loperand.dim());
	CHECK_EQUAL(roperand.dim(), 3, OperandSizeNotMatch,
		"BMM need 2D Tensor, but got %dD.", roperand.dim());
	CHECK_EQUAL(loperand.size(2), roperand.size(1), OperandSizeNotMatch,
		"BMM need lsize(2) and rsize(1) equal, but got size %d and %d.", loperand.size(2), roperand.size(1));
	// no check loperand.size(0) == roperand(0), which means allow broadcasting on batch dimension.
	return BMMExp<Dtype>(loperand, roperand);
}

} //namespace op
} // namespace el

#endif