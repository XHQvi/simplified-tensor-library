#ifndef EXPRESSION_EXPRESSION_H_
#define EXPRESSION_EXPRESSION_H_

#include <cmath>
#include <initializer_list>
#include "../base/type.h"

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
	Dtype eval(index_t idx) const {return OP::map(operand_.eval(idx));}
	// Dtype eval(initializer_list<index_t> ids) {return OP::map(operand_.eval(ids));}
};

template<typename OP, typename ROtype, typename LOtype, typename Dtype>
struct BinaryMapExp: public Exp<BinaryMapExp<OP, ROtype, LOtype, Dtype>, Dtype> {
	const ROtype& roperand_;
	const LOtype& loperand_;
	BinaryMapExp(const ROtype& roperand, const LOtype& loperand): roperand_(roperand), loperand_(loperand){}
	Dtype eval(index_t idx) const {return OP::map(roperand_.eval(idx), loperand_.eval(idx));}
	// Dtype eval(initializer_list<index_t> ids) {return OP::map(roperand_.eval(ids), loperand_.eval(ids));}
};

namespace op {
// unary operator
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
