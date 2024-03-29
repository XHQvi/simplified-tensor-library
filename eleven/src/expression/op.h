#ifndef EXPRESSION_OP_H_
#define EXPRESSION_OP_H_

#include "node.h"
#include "expression.h"
#include "../tensor/tensor_impl.h"

#include "operations/base_ops.h"
#include "operations/img2col.h"
#include "operations/matrix_multiply.h"
#include "operations/sigmoid.h"
#include "operations/relu.h"
#include "operations/nll_loss.h"
#include "operations/log_softmax.h"
#include "operations/max_pooling.h"
#include "operations/mean_reduce.h"
#include "operations/argmax.h"
#include "op_impl.h"

namespace el {

using op::operator+;
using op::operator-;
using op::operator*;

namespace op {
template<typename Dtype> Node<Dtype> node(const Tensor<Dtype>& tensor);
template<typename Dtype> Node<Dtype> node(const Tensor<Dtype>* tensor);

template<typename Dtype> MinusExp<Dtype> operator-(const Exp<Dtype>& operand);
template<typename Dtype> Node<Dtype> operator-(const Exp<Dtype>& operand);

template<typename Dtype> ReLUExp<Dtype> relu(const Exp<Dtype>& operand);
template<typename Dtype> Node<Dtype> relu(const Node<Dtype>& operand);

template<typename Dtype> SigmoidExp<Dtype> sigmoid(const Exp<Dtype>& operand);
template<typename Dtype> Node<Dtype> sigmoid(const Node<Dtype>& operand);

template<typename Dtype> MatrixTransposeExp<Dtype> transpose(const Exp<Dtype>& operand);
template<typename Dtype> Node<Dtype> transpose(const Node<Dtype>& operand);

template<typename Dtype> Img2ColExp<Dtype> img2col(const Exp<Dtype>& operand, 
								                   const std::pair<index_t, index_t>& kernel_size, 
								                   const std::pair<index_t, index_t>& stride, 
								                   const std::pair<index_t, index_t>& padding); 
template<typename Dtype> Node<Dtype> img2col(const Node<Dtype>& operand, 
								             const std::pair<index_t, index_t>& kernel_size, 
								             const std::pair<index_t, index_t>& stride, 
								             const std::pair<index_t, index_t>& padding); 


template<typename Dtype> AddExp<Dtype> operator+(const Exp<Dtype>& loperand, const Exp<Dtype>& roperand);
template<typename Dtype> Node<Dtype> operator+(const Node<Dtype>& loperand, const Node<Dtype>& roperand);

template<typename Dtype> SubExp<Dtype> operator-(const Exp<Dtype>& loperand, const Exp<Dtype>& roperand);
template<typename Dtype> Node<Dtype> operator-(const Node<Dtype>& loperand, const Node<Dtype>& roperand);

template<typename Dtype> MulExp<Dtype> operator*(const Exp<Dtype>& loperand, const Exp<Dtype>& roperand);
template<typename Dtype> Node<Dtype> operator*(const Node<Dtype>& loperand, const Node<Dtype>& roperand);

template<typename Dtype> MMExp<Dtype> mm(const Exp<Dtype>& loperand, const Exp<Dtype>& roperand);
template<typename Dtype> Node<Dtype> mm(const Node<Dtype>& loperand, const Node<Dtype>& roperand);

template<typename Dtype> BMMExp<Dtype> bmm(const Exp<Dtype>& loperand, const Exp<Dtype>& roperand);
template<typename Dtype> Node<Dtype> bmm(const Node<Dtype>& loperand, const Node<Dtype>& roperand);

template<typename Dtype> NLLLossExp<Dtype> nll_loss(const Exp<Dtype>& src, const Exp<int_t>& index);
template<typename Dtype> Node<Dtype> nll_loss(const Node<Dtype>& src, const Node<int_t>& index);

template<typename Dtype> LogSoftmaxExp<Dtype> log_softmax(const Exp<Dtype>& src);
template<typename Dtype> Node<Dtype> log_softmax(const Node<Dtype>& src);

template<typename Dtype> MaxPool2DExp<Dtype> maxpooling2d(const Exp<Dtype>& operand, const std::pair<index_t, index_t>& kernel_size);
template<typename Dtype> Node<Dtype> maxpooling2d(const Node<Dtype>& operand, const std::pair<index_t, index_t>& kernel_size);

template<typename Dtype> MeanReduceExp<Dtype> mean(const Exp<Dtype>& operand, index_t dim);
template<typename Dtype> Node<Dtype> mean(const Node<Dtype>& operand, index_t dim);

template<typename Dtype> ArgmaxExp<Dtype> argmax(const Exp<Dtype>& operand, index_t dim);
template<typename Dtype> Node<Dtype> argmax(const Node<Dtype>& operand, index_t dim);

}  // namespace op
}  // namespace el


#endif