#ifndef EXPRESSION_EXPRESSION_H_
#define EXPRESSION_EXPRESSION_H_

#include <cmath>
#include <initializer_list>
#include "../base/type.h"
#include "unary_exp.h"
#include "binary_exp.h"

namespace el {

template<typename Dtype>
struct Exp {
	virtual Dtype eval(index_t* ids) const = 0;
};

template<typename Dtype>
struct UnaryExp: public Exp<Dtype> {
	const Exp<Dtype>& operand_;
	UnaryExp(const Exp<Dtype>& operand): operand_(operand){}
};

template<typename Dtype>
struct BinaryExp: public Exp<Dtype> {
	const Exp<Dtype>& roperand_;
	const Exp<Dtype>& loperand_;
	BinaryExp(const Exp<Dtype>& roperand, const Exp<Dtype>& loperand): roperand_(roperand), loperand_(loperand){}
};

} // namespace el


#endif
