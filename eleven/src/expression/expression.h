#ifndef EXPRESSION_EXPRESSION_H_
#define EXPRESSION_EXPRESSION_H_

#include <cmath>
#include <initializer_list>
#include "../base/type.h"
#include "unary_exp.h"
#include "binary_exp.h"

namespace el {

template<typename SubType, typename Dtype>
struct Exp {
	inline const SubType& self() const {return *static_cast<const SubType*>(this);}
};

// OP: operator; Otype: operand type; Dtype: data type;
template<typename OP, typename Otype, typename Dtype>
struct UnaryMapExp: public Exp<UnaryMapExp<OP, Otype, Dtype>, Dtype> {
	const Otype& operand_;
	UnaryMapExp(const Otype& operand): operand_(operand){}
	Dtype eval(index_t* ids) const {return OP::map(operand_.eval(ids));}
	// Dtype eval(initializer_list<index_t> ids) {return OP::map(operand_.eval(ids));}
};

// ROtype/LOtype: right/left operand type;
template<typename OP, typename ROtype, typename LOtype, typename Dtype>
struct BinaryMapExp: public Exp<BinaryMapExp<OP, ROtype, LOtype, Dtype>, Dtype> {
	const ROtype& roperand_;
	const LOtype& loperand_;
	BinaryMapExp(const ROtype& roperand, const LOtype& loperand): roperand_(roperand), loperand_(loperand){}
	Dtype eval(index_t* ids) const {return OP::map(roperand_.eval(ids), loperand_.eval(ids));}
	// Dtype eval(initializer_list<index_t> ids) {return OP::map(roperand_.eval(ids), loperand_.eval(ids));}
};
} // namespace el


#endif
