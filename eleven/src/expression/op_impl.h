#ifndef EXPRESSION_OP_IMPL_H_
#define EXPRESSION_OP_IMPL_H_

#include "op.h"

namespace el {
namespace op {

template<typename Dtype>
inline Node<Dtype> node(const Tensor<Dtype>& tensor) {
	return Node<Dtype>(new Tensor<Dtype>(tensor));
}

template<typename Dtype>
inline Node<Dtype> node(const Tensor<Dtype>* tensor) {
	return Node<Dtype>(tensor);
}

template<typename Dtype>
inline MinusExp<Dtype> operator-(const Exp<Dtype>& operand) {
	return MinusExp<Dtype>(operand);
}
template<typename Dtype>
inline Node<Dtype> operator-(const Node<Dtype>& operand) {
	return Node<Dtype>(new MinusExp<Dtype>(operand.get_exp_ptr()));
}

template<typename Dtype>
inline ReLUExp<Dtype> relu(const Exp<Dtype>& operand) {
	return ReLUExp<Dtype>(operand);
}
template<typename Dtype>
inline Node<Dtype> relu(const Node<Dtype>& operand) {
	return Node<Dtype>(new ReLUExp<Dtype>(operand.get_exp_ptr()));
}

template<typename Dtype>
inline SigmoidExp<Dtype> sigmoid(const Exp<Dtype>& operand) {
	return SigmoidExp<Dtype>(operand);
}
template<typename Dtype>
inline Node<Dtype> sigmoid(const Node<Dtype>& operand) {
	return Node<Dtype>(new SigmoidExp<Dtype>(operand.get_exp_ptr()));
}

template<typename Dtype>
inline MatrixTransposeExp<Dtype> transpose(const Exp<Dtype>& operand) {
	CHECK_EQUAL(operand.dim(), 2, DimNotMatch,
		"Matrix Transpose expect 2D matrix, but got %dD tensor", operand.dim());
	return MatrixTransposeExp<Dtype>(operand);
}
template<typename Dtype>
inline Node<Dtype> transpose(const Node<Dtype>& operand) {
	return Node<Dtype>(new MatrixTransposeExp<Dtype>(operand.get_exp_ptr()));
}

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
		kernel_size.first, kernel_size.second,
		stride.first, stride.second);
	return ret;
}
template<typename Dtype>
inline Node<Dtype> img2col(const Node<Dtype>& operand, 
						   const std::pair<index_t, index_t>& kernel_size, 
						   const std::pair<index_t, index_t>& stride, 
						   const std::pair<index_t, index_t>& padding) {
	CHECK_EQUAL(operand.dim(), 4, DimNotMatch,
		"Img2ColExp expect 4D tensor:(b, c, h, w), but got %dD tensor", operand.dim());
	Img2ColExp<Dtype>* ret = new Img2ColExp<Dtype>(operand.get_exp_ptr(), kernel_size, stride, padding);
	CHECK_TRUE(ret->out_size(0) > 0, OperandSizeNotMatch,
		"Can't convolve on image(%d, %d) because of too big kernel size(%d, %d) or stride(%d, %d)", 
		operand.size(2), operand.size(3),
		kernel_size.first, kernel_size.second,
		stride.first, stride.second);
	return Node<Dtype>(ret);
}

template<typename Dtype>
inline AddExp<Dtype> operator+(const Exp<Dtype>& loperand, const Exp<Dtype>& roperand) {
	CHECK_BROADCAST(loperand, roperand);
	return AddExp<Dtype>(loperand, roperand);
}
template<typename Dtype>
inline Node<Dtype> operator+(const Node<Dtype>& loperand, const Node<Dtype>& roperand) {
	CHECK_BROADCAST(loperand, roperand);
	return Node<Dtype>(new AddExp<Dtype>(loperand.get_exp_ptr(), roperand.get_exp_ptr()));
}

template<typename Dtype>
inline SubExp<Dtype> operator-(const Exp<Dtype>& loperand, const Exp<Dtype>& roperand) {
	CHECK_BROADCAST(loperand, roperand);
	return SubExp<Dtype>(loperand, roperand);
}
template<typename Dtype>
inline Node<Dtype> operator-(const Node<Dtype>& loperand, const Node<Dtype>& roperand) {
	CHECK_BROADCAST(loperand, roperand);
	return Node<Dtype>(new SubExp<Dtype>(loperand.get_exp_ptr(), roperand.get_exp_ptr()));
}

template<typename Dtype>
inline MulExp<Dtype> operator*(const Exp<Dtype>& loperand, const Exp<Dtype>& roperand) {
	CHECK_BROADCAST(loperand, roperand);
	return MulExp<Dtype>(loperand, roperand);
}
template<typename Dtype>
inline Node<Dtype> operator*(const Node<Dtype>& loperand, const Node<Dtype>& roperand) {
	CHECK_BROADCAST(loperand, roperand);
	return Node<Dtype>(new MulExp<Dtype>(loperand.get_exp_ptr(), roperand.get_exp_ptr()));
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
	return Node<Dtype>(new MMExp<Dtype>(loperand.get_exp_ptr(), roperand.get_exp_ptr()));
}

template<typename Dtype>
inline BMMExp<Dtype> bmm(const Exp<Dtype>& loperand, const Exp<Dtype>& roperand) {
	CHECK_EQUAL(loperand.dim(), 3, OperandSizeNotMatch,
		"BMM need 3D Tensor, but got %dD.", loperand.dim());
	CHECK_EQUAL(roperand.dim(), 3, OperandSizeNotMatch,
		"BMM need 3D Tensor, but got %dD.", roperand.dim());
	CHECK_EQUAL(loperand.size(2), roperand.size(1), OperandSizeNotMatch,
		"BMM need lsize(2) and rsize(1) equal, but got size %d and %d.", loperand.size(2), roperand.size(1));
	// no check for loperand.size(0) == roperand(0), which means allow broadcasting on batch dimension.
	return BMMExp<Dtype>(loperand, roperand);
}
template<typename Dtype>
inline Node<Dtype> bmm(const Node<Dtype>& loperand, const Node<Dtype>& roperand) {
	CHECK_EQUAL(loperand.dim(), 3, OperandSizeNotMatch,
		"BMM need 3D Tensor, but got %dD.", loperand.dim());
	CHECK_EQUAL(roperand.dim(), 3, OperandSizeNotMatch,
		"BMM need 3D Tensor, but got %dD.", roperand.dim());
	CHECK_EQUAL(loperand.size(2), roperand.size(1), OperandSizeNotMatch,
		"BMM need lsize(2) and rsize(1) equal, but got size %d and %d.", loperand.size(2), roperand.size(1));
	// no check for loperand.size(0) == roperand(0), which means allow broadcasting on batch dimension.
	return Node<Dtype>(new BMMExp<Dtype>(loperand.get_exp_ptr(), roperand.get_exp_ptr()));
}

template<typename Dtype> NLLLossExp<Dtype> nll_loss(const Exp<Dtype>& src, const Exp<int_t>& index) {
	CHECK_EQUAL(src.dim(), 2, OperandSizeNotMatch,
		"BatchIndex is only used on 2D tensor as src, but got %dD tensor", src.dim());
	CHECK_EQUAL(index.dim(), 1, OperandSizeNotMatch,
		"BatchIndex is only used on 1D tensor as index, but got %dD tensor", index.dim());
	return NLLLossExp<Dtype>(src, index);
}
template<typename Dtype> Node<Dtype> nll_loss(const Node<Dtype>& src, const Node<int_t>& index) {
	CHECK_EQUAL(src.dim(), 2, OperandSizeNotMatch,
		"BatchIndex is only used on 2D tensor as src, but got %dD tensor", src.dim());
	CHECK_EQUAL(index.dim(), 1, OperandSizeNotMatch,
		"BatchIndex is only used on 1D tensor as index, but got %dD tensor", index.dim());
	return Node<Dtype>(new NLLLossExp<Dtype>(src.get_exp_ptr(), index.get_exp_ptr()));
}

template<typename Dtype> LogSoftmaxExp<Dtype> log_softmax(const Exp<Dtype>& src) {
	CHECK_EQUAL(src.dim(), 2, OperandSizeNotMatch,
		"log_softmax is only implemented for 1D tensor, but got %dD tensor.", src.dim());
	return LogSoftmaxExp<Dtype>(src);
}
template<typename Dtype> Node<Dtype> log_softmax(const Node<Dtype>& src) {
	CHECK_EQUAL(src.dim(), 2, OperandSizeNotMatch,
		"log_softmax is only implemented for tensor with shape (batch_size, num_cls), but got %dD tensor.", src.dim());
	return Node<Dtype>(new LogSoftmaxExp<Dtype>(src.get_exp_ptr()));	
}

}  // namespace op
}  // namespace el


#endif