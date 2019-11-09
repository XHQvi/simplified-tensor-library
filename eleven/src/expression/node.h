#ifndef EXPRESSION_NODE_H_
#define EXPRESSION_NODE_H_

#include <memory>
#include "expression.h"

namespace el {

template<typename Dtype> class Tensor;

template<typename Dtype>
class Node {
public:
	// constructors
	explicit Node(Exp<Dtype>* exp_ptr);
	explicit Node(Tensor<Dtype>* exp_ptr);
	// shortcut to exp
	void backward(void) const;
	index_t dim(void) const;
	index_t size(index_t idx) const;
	// node's method
	bool contain_tensor(void) const;
	const Tensor<Dtype>& get_tensor(void) const;
	const Exp<Dtype>& get_exp(void) const;
	const std::shared_ptr<Exp<Dtype>>& get_exp_ptr(void) const;
	template<template<typename Dtype1> class ExpType> const ExpType<Dtype>& get(void) const;
private:
	const std::shared_ptr<Exp<Dtype>> exp_ptr_;
	const index_t version_;
};

template<typename Dtype>
Node<Dtype>::Node(Exp<Dtype>* exp_ptr)
	: exp_ptr_(exp_ptr), version_(-1) {}

template<typename Dtype>
Node<Dtype>::Node(Tensor<Dtype>* exp_ptr)
	: exp_ptr_(exp_ptr), version_(exp_ptr->version()) {}

template<typename Dtype>
inline bool Node<Dtype>::contain_tensor(void) const {return version_ >= 0;}

// TODO: ONLY TENSOR NODE'S BACKWARD FUNCTION CAN BE CALLED.
// ONLY WHEN NODE'S VERSION IS EQUAL TO TENSOR'S VERSION, THIS FUNCTION CAN BE CALLED.
template<typename Dtype>
inline void Node<Dtype>::backward(void) const {
	exp_ptr_->backward();
}

template<typename Dtype>
inline index_t Node<Dtype>::dim(void) const {return exp_ptr_->dim();}

template<typename Dtype>
inline index_t Node<Dtype>::size(index_t idx) const {return exp_ptr_->size(idx);}

template<typename Dtype>
	template<template<typename Dtype1> class ExpType>
inline const ExpType<Dtype>& Node<Dtype>::get(void) const {
	return *static_cast<ExpType<Dtype>*>(exp_ptr_.get());
}

template<typename Dtype>
inline const Tensor<Dtype>& Node<Dtype>::get_tensor(void) const {
	CHECK_TRUE(contain_tensor(), NodeTypeWrong,
		"Can't get a tensor reference from a node not containing a tensor");
	return get<Tensor>();
}

template<typename Dtype>
inline const Exp<Dtype>& Node<Dtype>::get_exp(void) const {
	return *exp_ptr_.get();
}

template<typename Dtype>
inline const std::shared_ptr<Exp<Dtype>>& Node<Dtype>::get_exp_ptr(void) const {
	return exp_ptr_;
}

}  // namespace el

#endif