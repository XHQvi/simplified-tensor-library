#ifndef EXPRESSION_EXPRESSION_H_
#define EXPRESSION_EXPRESSION_H_

#include <cmath>
#include <initializer_list>
#include "../base/type.h"
#include "../base/exception.h"
#include "unary_exp.h"
#include "binary_exp.h"

namespace el {

template<typename Dtype>
struct Exp {
	virtual Dtype eval(index_t* ids) const = 0;
	virtual index_t dim(void) const = 0;
	virtual index_t size(index_t idx) const = 0;
	template<typename Dtype1>friend std::ostream operator<< (std::ostream& out, const Exp<Dtype1>& exp);
};

template<typename Dtype>
struct UnaryExp: public Exp<Dtype> {
	const Exp<Dtype>& operand_;
	UnaryExp(const Exp<Dtype>& operand): operand_(operand){}
	virtual index_t dim(void) const {return this->operand_.dim();}
	virtual index_t size(index_t idx) const {return this->operand_.size(idx);}
};

template<typename Dtype>
struct BinaryExp: public Exp<Dtype> {
	const Exp<Dtype>& loperand_;
	const Exp<Dtype>& roperand_;
	BinaryExp(const Exp<Dtype>& loperand, const Exp<Dtype>& roperand): roperand_(roperand), loperand_(loperand){}
	virtual index_t dim(void) const {return this->roperand_.dim();}
	virtual index_t size(index_t idx) const {return std::max(this->roperand_.size(idx), this->loperand_.size(idx));}
};

} // namespace el


#endif
