#ifndef EXPRESSION_EXPRESSION_H_
#define EXPRESSION_EXPRESSION_H_

#include <cmath>
#include <memory>
#include <initializer_list>
#include "../utils/base.h"

namespace el {
template<typename Dtype> class ConstExptr;
template<typename Dtype> class Node;

template<typename Dtype>
class Exp {
private:
	mutable index_t refcount_ = 0;
	mutable index_t gradcount_ = 0;
	virtual void backward(const Exp<Dtype>& grad) const = 0;
public:
	virtual Dtype eval(index_t* ids) const = 0;
	virtual index_t dim(void) const = 0;
	virtual index_t size(index_t idx) const = 0;
	virtual ~Exp() {};
	friend class ConstExptr<Dtype>;
	friend class Node<Dtype>;
};
}  // namespace el

#include "const_exptr.h"

namespace el {

template<typename Dtype>
class UnaryExp: public Exp<Dtype> {
public:
	// TODO: I'm considering about deletion of the constructor using reference.
	// Because It may cause ConstExptr point to a tensor on stack storage.
	// Deconstructor of this tensor may be called when the Exp is deconstructed,
	// or the stack storage is released. Anyway, it's dangerous.
	UnaryExp(const Exp<Dtype>& operand): operand_(&operand, true){}
	UnaryExp(const Exp<Dtype>* operand): operand_(operand, true) {}
	virtual index_t dim(void) const {return this->operand_->dim();}
	virtual index_t size(index_t idx) const {return this->operand_->size(idx);}
protected:
	const ConstExptr<Dtype> operand_;
};

template<typename Dtype>
struct BinaryExp: public Exp<Dtype> {
public:
	BinaryExp(const Exp<Dtype>& loperand, const Exp<Dtype>& roperand)
		    : roperand_(&roperand, true), loperand_(&loperand, true){}
	BinaryExp(const Exp<Dtype>* loperand, const Exp<Dtype>* roperand)
			: roperand_(roperand, true), loperand_(loperand, true){}
	virtual index_t dim(void) const {return this->roperand_->dim();}
	virtual index_t size(index_t idx) const {return std::max(this->roperand_->size(idx), this->loperand_->size(idx));}
protected:
	const ConstExptr<Dtype> loperand_;
	const ConstExptr<Dtype> roperand_;
};
} // namespace el


#endif
