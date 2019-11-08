#ifndef EXPRESSION_OP_IMPL_H_
#define EXPRESSION_OP_IMPL_H_

#include "expression.h"

namespace el {
namespace op {

template<typename Dtype>
inline AbsExp<Dtype> abs(const Exp<Dtype>& operand) {
	return AbsExp<Dtype>(operand);
}

template<typename Dtype>
inline SigmoidExp<Dtype> sigmoid(const Exp<Dtype>& operand) {
	return SigmoidExp<Dtype>(operand);
}

template<typename Dtype>
Img2ColExp<Dtype> img2col(const Exp<Dtype>& operand, 
								 const std::pair<index_t, index_t>& kernel_size, 
								 const std::pair<index_t, index_t>& stride, 
								 const std::pair<index_t, index_t>& padding) {
	// CHECK_DIM_MATCH(operand.dim(), 4);  // batch_size, c, h, w
	return Img2ColExp<Dtype>(operand, kernel_size, stride, padding);
}

template<typename Dtype>
inline AddExp<Dtype> operator+(const Exp<Dtype>& loperand, const Exp<Dtype>& roperand) {
	CHECK_BROADCAST(loperand, roperand);
	return AddExp<Dtype>(loperand, roperand);
}


template<typename Dtype>
inline SubExp<Dtype> operator-(const Exp<Dtype>& loperand, const Exp<Dtype>& roperand) {
	CHECK_BROADCAST(loperand, roperand);
	return SubExp<Dtype>(loperand, roperand);
}

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


}  // namespace op
}  // namespace el


#endif