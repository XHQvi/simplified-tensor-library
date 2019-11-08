#ifndef EXPRESSION_NODE_H_
#define EXPRESSION_NODE_H_

#include <memory>
#include "expression.h"

namespace el {

template<typename Dtype> class Tensor;

template<typename Dtype>
class Node {
public:
	void backward(void) const;
	explicit Node(Exp<Dtype>* exp_ptr);
	explicit Node(Tensor<Dtype>* exp_ptr);
	index_t dim(void) const;
	index_t size(index_t idx) const;
	const Tensor<Dtype>& get(void) const;
	const Exp<Dtype>& get_exp(void) const;
private:
	const std::shared_ptr<Exp<Dtype>> exp_ptr_;
	bool is_tensor_;
};

template<typename Dtype>
Node<Dtype>::Node(Exp<Dtype>* exp_ptr)
	: exp_ptr_(exp_ptr), is_tensor_(false) {}

template<typename Dtype>
Node<Dtype>::Node(Tensor<Dtype>* exp_ptr)
	: exp_ptr_(exp_ptr), is_tensor_(true) {}

template<typename Dtype>
inline void Node<Dtype>::backward(void) const {
	exp_ptr_->backward();
}

template<typename Dtype>
inline index_t Node<Dtype>::dim(void) const {
	return exp_ptr_->dim();
}

template<typename Dtype>
inline index_t Node<Dtype>::size(index_t idx) const {
	return exp_ptr_->size(idx);
}

template<typename Dtype>
inline const Tensor<Dtype>& Node<Dtype>::get(void) const {
	CHECK_TRUE(is_tensor_, NodeTypeWrong,
		"Can't get a tensor reference from a node not containing a tensor");
	return *static_cast<Tensor<Dtype>*>(exp_ptr_.get());
}

template<typename Dtype>
inline const Exp<Dtype>& Node<Dtype>::get_exp(void) const {
	return *exp_ptr_.get();
}
}  // namespace el

#endif