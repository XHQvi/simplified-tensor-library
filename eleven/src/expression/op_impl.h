#ifndef EXPRESSION_OP_IMPL_H_
#define EXPRESSION_OP_IMPL_H_

#include "expression.h"
#include "node.h"

namespace el {
namespace op {

template<typename Dtype>
inline Node<Dtype> node(const Tensor<Dtype>& tensor) {
	return Node<Dtype>(new Tensor<Dtype>(tensor));
}

template<typename Dtype>
inline AbsExp<Dtype> abs(const Exp<Dtype>& operand) {
	return AbsExp<Dtype>(operand);
}
template<typename Dtype>
inline Node<Dtype> abs(const Node<Dtype>& operand) {
	return Node<Dtype>(new AbsExp<Dtype>(operand.get_exp()));
}

template<typename Dtype>
inline SigmoidExp<Dtype> sigmoid(const Exp<Dtype>& operand) {
	return SigmoidExp<Dtype>(operand);
}
template<typename Dtype>
inline Node<Dtype> sigmoid(const Node<Dtype>& operand) {
	return Node<Dtype>(new SigmoidExp<Dtype>(operand.get_exp()));
}


// TODO:
// check operand's size matchs expectation.
template<typename Dtype>
inline Img2ColExp<Dtype> img2col(const Exp<Dtype>& operand, 
								 const std::pair<index_t, index_t>& kernel_size, 
								 const std::pair<index_t, index_t>& stride, 
								 const std::pair<index_t, index_t>& padding) {
	CHECK_EQUAL(operand.dim(), 4, DimNotMatch,
		"Img2ColExp expect 4D tensor:(b, c, h, w), but got %dD tensor", operand.dim());
	Img2ColExp<Dtype> ret (operand, kernel_size, stride, padding);
	CHECK_TRUE(ret.out_size(0) > 0, OperandSizeNotMatch,
		"Can't convolve on image(%d, %d) because of too big kernel size(%d, %d) or stride(%d, %d)", 
		operand.size(2), operand.size(3),
		ret.kernel_size_.first, ret.kernel_size_.second,
		ret.stride_.first, ret.stride_.second);
	return ret;
}
template<typename Dtype>
inline Node<Dtype> img2col(const Node<Dtype>& operand, 
						   const std::pair<index_t, index_t>& kernel_size, 
						   const std::pair<index_t, index_t>& stride, 
						   const std::pair<index_t, index_t>& padding) {
	CHECK_EQUAL(operand.dim(), 4, DimNotMatch,
		"Img2ColExp expect 4D tensor:(b, c, h, w), but got %dD tensor", operand.dim());
	Img2ColExp<Dtype> ret (operand, kernel_size, stride, padding);
	CHECK_TRUE(ret.out_size(0) > 0, OperandSizeNotMatch,
		"Can't convolve on image(%d, %d) because of too big kernel size(%d, %d) or stride(%d, %d)", 
		operand.size(2), operand.size(3),
		ret.kernel_size_.first, ret.kernel_size_.second,
		ret.stride_.first, ret.stride_.second);
	return ret;
}

template<typename Dtype>
inline AddExp<Dtype> operator+(const Exp<Dtype>& loperand, const Exp<Dtype>& roperand) {
	CHECK_BROADCAST(loperand, roperand);
	return AddExp<Dtype>(loperand, roperand);
}
template<typename Dtype>
inline Node<Dtype> operator+(const Node<Dtype>& loperand, const Node<Dtype>& roperand) {
	CHECK_BROADCAST(loperand, roperand);
	return Node<Dtype>(new AddExp<Dtype>(loperand.get_exp(), roperand.get_exp()));
}

template<typename Dtype>
inline SubExp<Dtype> operator-(const Exp<Dtype>& loperand, const Exp<Dtype>& roperand) {
	CHECK_BROADCAST(loperand, roperand);
	return SubExp<Dtype>(loperand, roperand);
}
template<typename Dtype>
inline Node<Dtype> operator-(const Node<Dtype>& loperand, const Node<Dtype>& roperand) {
	CHECK_BROADCAST(loperand, roperand);
	return Node<Dtype>(new SubExp<Dtype>(loperand.get_exp(), roperand.get_exp()));
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
inline Node<Dtype> mm(const Node<Dtype>& loperand, const Node<Dtype>& roperand) {
	CHECK_EQUAL(loperand.dim(), 2, OperandSizeNotMatch,
		"MM need 2D Tensor, but got %dD.", loperand.dim());
	CHECK_EQUAL(roperand.dim(), 2, OperandSizeNotMatch,
		"MM need 2D Tensor, but got %dD.", roperand.dim());
	CHECK_EQUAL(loperand.size(1), roperand.size(0), OperandSizeNotMatch,
		"MM need lsize(1) and rsize(0) equal, but got size %d and %d.", loperand.size(1), roperand.size(0));
	return Node<Dtype>(new MMExp<Dtype>(loperand.get_exp(), roperand.get_exp()));
}

template<typename Dtype>
inline BMMExp<Dtype> bmm(const Exp<Dtype>& loperand, const Exp<Dtype>& roperand) {
	CHECK_EQUAL(loperand.dim(), 3, OperandSizeNotMatch,
		"BMM need 2D Tensor, but got %dD.", loperand.dim());
	CHECK_EQUAL(roperand.dim(), 3, OperandSizeNotMatch,
		"BMM need 2D Tensor, but got %dD.", roperand.dim());
	CHECK_EQUAL(loperand.size(2), roperand.size(1), OperandSizeNotMatch,
		"BMM need lsize(2) and rsize(1) equal, but got size %d and %d.", loperand.size(2), roperand.size(1));
	// no check for loperand.size(0) == roperand(0), which means allow broadcasting on batch dimension.
	return BMMExp<Dtype>(loperand, roperand);
}
template<typename Dtype>
inline Node<Dtype> bmm(const Node<Dtype>& loperand, const Node<Dtype>& roperand) {
	CHECK_EQUAL(loperand.dim(), 3, OperandSizeNotMatch,
		"BMM need 2D Tensor, but got %dD.", loperand.dim());
	CHECK_EQUAL(roperand.dim(), 3, OperandSizeNotMatch,
		"BMM need 2D Tensor, but got %dD.", roperand.dim());
	CHECK_EQUAL(loperand.size(2), roperand.size(1), OperandSizeNotMatch,
		"BMM need lsize(2) and rsize(1) equal, but got size %d and %d.", loperand.size(2), roperand.size(1));
	// no check for loperand.size(0) == roperand(0), which means allow broadcasting on batch dimension.
	return Node<Dtype>(new BMMExp<Dtype>(loperand.get_exp(), roperand.get_exp()));
}


}  // namespace op
}  // namespace el


#endif