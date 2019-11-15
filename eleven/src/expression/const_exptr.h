#ifndef EXPRESSION_EXPTR_H_
#define EXPRESSION_EXPTR_H_

#include "../utils/base.h"

namespace el {
template<typename Dtype> class Exp;

template<typename Dtype>
class ConstExptr {
public:
	// member function
	explicit ConstExptr(void);
	explicit ConstExptr(const Exp<Dtype>* ptr, bool with_grad);
	explicit ConstExptr(const ConstExptr& other, bool with_grad);
	explicit ConstExptr(const ConstExptr& other);
	~ConstExptr();
	// modifiers
	void reset(const Exp<Dtype>* ptr, bool with_grad);
	// observers
	const Exp<Dtype>* get(void) const;
	const Exp<Dtype>& operator*(void) const;
	const Exp<Dtype>* operator->(void) const;
	long use_count(void) const;
	bool unique(void) const;
	explicit operator bool(void) const;
	// wrap the backward function of expression
	void backward(const Exp<Dtype>& grad) const;
	// static method
	static void make_uncontrol(const Exp<Dtype>& exp);
	static bool grad_ready(const Exp<Dtype>& exp);
private:
	const Exp<Dtype>* ptr_;
	void increment_counters(bool with_grad);
	void decrement_refcount(void);
};

template<typename Dtype>
ConstExptr<Dtype>::ConstExptr(void) :ptr_(nullptr) {}

template<typename Dtype>
ConstExptr<Dtype>::ConstExptr(const Exp<Dtype>* ptr, bool with_grad) :ptr_(ptr) {
	increment_counters(with_grad);
}

template<typename Dtype>
ConstExptr<Dtype>::ConstExptr(const ConstExptr& other, bool with_grad) :ptr_(other.ptr_) {
	increment_counters(with_grad);
}

template<typename Dtype>
ConstExptr<Dtype>::ConstExptr(const ConstExptr& other): ptr_(other.ptr_) {
	increment_counters(false);
}

template<typename Dtype>
ConstExptr<Dtype>::~ConstExptr() {
	decrement_refcount();
}

template<typename Dtype>
inline void ConstExptr<Dtype>::increment_counters(bool with_grad) {
	if(ptr_ != nullptr) {
		ptr_->refcount_ ++;
		if(with_grad)
			ptr_->gradcount_ ++;
	}
}

template<typename Dtype>
inline void ConstExptr<Dtype>::decrement_refcount(void) {
	if(ptr_ != nullptr) {
		ptr_->refcount_ --;
		if(ptr_->refcount_ <= 0)
			delete ptr_;
	}
}

template<typename Dtype>
inline void ConstExptr<Dtype>::reset(const Exp<Dtype>* ptr, bool with_grad) {
	decrement_refcount();
	ptr_ = ptr;
	increment_counters(with_grad);
}

template<typename Dtype>
inline const Exp<Dtype>* ConstExptr<Dtype>::get(void) const {return ptr_;}

template<typename Dtype>
inline const Exp<Dtype>& ConstExptr<Dtype>::operator*(void) const {return *ptr_;}

template<typename Dtype>
inline const Exp<Dtype>* ConstExptr<Dtype>::operator->(void) const {return ptr_;}

template<typename Dtype>
inline long ConstExptr<Dtype>::use_count(void) const {return ptr_->refcount_;}

template<typename Dtype>
inline bool ConstExptr<Dtype>::unique(void) const {return ptr_->refcount_ == 1;}

template<typename Dtype>
inline ConstExptr<Dtype>::operator bool(void) const {return ptr_ != nullptr;}

template<typename Dtype>
inline void ConstExptr<Dtype>::make_uncontrol(const Exp<Dtype>& exp) {
	exp.refcount_ = INDEX_MAX / 2;
	exp.gradcount_ = INDEX_MAX / 2;
}

template<typename Dtype>
inline bool ConstExptr<Dtype>::grad_ready(const Exp<Dtype>& exp) {
	return exp.gradcount_ == 0;
}

template<typename Dtype>
inline void ConstExptr<Dtype>::backward(const Exp<Dtype>& grad) const {
	ptr_->gradcount_ --;
	ptr_->backward(grad);
}

}  // namespace el
#endif
