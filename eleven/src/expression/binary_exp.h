#ifndef EXPRESSION_BINARY_EXP_H_
#define EXPRESSION_BINARY_EXP_H_

#include "expression.h"
#include <iostream>

namespace el {

template<typename Dtype> struct Exp;
template<typename Dtype> struct BinaryExp;

namespace op {

template<typename Dtype>
struct AddExp: public BinaryExp<Dtype> {
	AddExp(const Exp<Dtype>& loperand, const Exp<Dtype>& roperand): BinaryExp<Dtype>(loperand, roperand){}
	Dtype eval(index_t* ids) const {return this->loperand_.eval(ids) + this->roperand_.eval(ids);}
	void backward(void) const {
		std::cout << "add backward" << std::endl;
		this->loperand_.backward();
		this->roperand_.backward();
	}
};

template<typename Dtype>
struct SubExp: public BinaryExp<Dtype> {
	SubExp(const Exp<Dtype>& loperand, const Exp<Dtype>& roperand): BinaryExp<Dtype>(loperand, roperand){}
	Dtype eval(index_t* ids) const {return this->loperand_.eval(ids) - this->roperand_.eval(ids);}
};

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
	void backward(void) const {
		std::cout << "bmm backward" << std::endl;
		this->loperand_.backward();
		this->roperand_.backward();
	}
};

} //namespace op
} // namespace el

#endif