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
// private:
private:
	mutable index_t refcount_ = 0;
	mutable index_t gradcount_ = 0;
	virtual void backward(const Exp<Dtype>& grad) const = 0;
public:
	virtual Dtype eval(index_t* ids) const = 0;
	virtual index_t dim(void) const = 0;
	virtual index_t size(index_t idx) const = 0;
	virtual bool requires_grad(void) const = 0;
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
	// The constructor taking a reference as parameter may cause the ConstExptr point to a exp on stack storage. It's 
	// dangerous just like making a shared_ptr point to a object on stack storage. Deconstructor of this exp may be 
	// called twice, when stack storage is freed or this UnaryExp is deconstructed. To avoid this, I use 
	// ConstExptr::make_uncontrol, which checks whether the exp has been bound to any ConstExptr, then if not, makes the
	// exp out of control of ConstExptr. (If the exp has been bound to ConstExptrs, we intend to believe the exp is from
	// new operator, and do nothing.)
	// 
	// Also, the constructor taking a pointer may cause the same problem, if using it improperly. But I think a pointer is
	// always with a hint that "my storage is from new operator instead of static storage". Anyway, pay attention to that.
	//
	// The parameter, with_grad, of ConstExptr's constructor decides whether gradient should be passed to operand. If 
	// with_grad is false or the operand's requires_grad is false, the gradient shouldn't. As mentioned above, the 
	// constructor taking a reference as parameter ofter means the parameter is on stack storage, and it's not safe to add
	// a exp on stack storage into a computation graph. Because we need the same one exp when we backward gradient, but the
	// exp on stack storage may have be freed, consifering that forward and backward may happen in different scopes. This is
	// the reason why the two constructors using different bool value for the parameter with_grad.
	UnaryExp(const Exp<Dtype>& operand) {
		ConstExptr<Dtype>::make_uncontrol(operand);
		operand_.reset(&operand, false);
	}
	UnaryExp(const Exp<Dtype>* operand)
		: operand_(operand, /*with_grad=*/true) {}

	virtual index_t dim(void) const {return this->operand_->dim();}
	virtual index_t size(index_t idx) const {return this->operand_->size(idx);}
	virtual bool requires_grad(void) const final {return operand_.requires_grad();}
protected:
	ConstExptr<Dtype> operand_;
};

template<typename Dtype>
struct BinaryExp: public Exp<Dtype> {
public:
	BinaryExp(const Exp<Dtype>& loperand, const Exp<Dtype>& roperand) {
		ConstExptr<Dtype>::make_uncontrol(loperand);
		ConstExptr<Dtype>::make_uncontrol(roperand);
		loperand_.reset(&loperand, false);
		roperand_.reset(&roperand, false);
	}
	BinaryExp(const Exp<Dtype>* loperand, const Exp<Dtype>* roperand)
			: loperand_(loperand, /*with_grad=*/true), roperand_(roperand, /*with_grad=*/true) {}

	virtual index_t dim(void) const {return this->roperand_->dim();}
	virtual index_t size(index_t idx) const {return std::max(this->roperand_->size(idx), this->loperand_->size(idx));}
	virtual bool requires_grad(void) const final {return loperand_.requires_grad() || roperand_.requires_grad();}
protected:
	ConstExptr<Dtype> loperand_;
	ConstExptr<Dtype> roperand_;
};

template<typename Dtype>
struct ConstantExp: public Exp<Dtype> {
	explicit ConstantExp(Dtype value, index_t dim): value_(value), dim_(dim) {}
	Dtype eval(index_t* ids) const {return value_;}
	index_t dim(void) const {return dim_;}
	index_t size(index_t idx) const {return 1;}
	bool requires_grad(void) const {return false;}
	void backward(const Exp<Dtype>& grad) const {
		THROW_ERROR(NotImplementError, "Can't call backward for a constant.");
	}
private:
	Dtype value_;
	index_t dim_;
};

} // namespace el


#endif
