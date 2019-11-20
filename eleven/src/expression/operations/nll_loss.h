#ifndef EXPRESSION_OPERATIONS_NLL_LOSS_H_
#define EXPRESSION_OPERATIONS_NLL_LOSS_H_

#include "../expression.h"

namespace el {
namespace op {

template<typename Dtype>
struct NLLLossExp: public Exp<Dtype> {
	NLLLossExp(const Exp<Dtype>& src, const Exp<int_t>& index);
	NLLLossExp(const Exp<Dtype>* src, const Exp<int_t>* index);

	index_t dim(void) const;
	index_t size(index_t idx) const;
	bool requires_grad(void) const;
	Dtype eval(index_t* ids) const;
	void backward(const Exp<Dtype>& grad) const;
private:
	ConstExptr<Dtype> src_;
	ConstExptr<int_t> index_;
	
	struct GradExp: public Exp<Dtype> {
	public:	
		GradExp(const Exp<Dtype>& grad, const Exp<Dtype>& src, const Exp<int_t>& index);
		index_t dim(void) const;
		index_t size(index_t idx) const;
		bool requires_grad(void) const;
		Dtype eval(index_t* ids) const;
		void backward(const Exp<Dtype>& grad) const;
	private:
		ConstExptr<Dtype> grad_;
		ConstExptr<Dtype> src_;
		ConstExptr<int_t> index_;
	};
};

template<typename Dtype>
NLLLossExp<Dtype>::NLLLossExp(const Exp<Dtype>& src, const Exp<int_t>& index) {
	ConstExptr<Dtype>::make_uncontrol(src);
	ConstExptr<int_t>::make_uncontrol(index);
	src_.reset(&src, false);
	index_.reset(&index, false);
}

template<typename Dtype>
NLLLossExp<Dtype>::NLLLossExp(const Exp<Dtype>* src, const Exp<int_t>* index)
	: src_(src, true), index_(index, false) {}

template<typename Dtype>
inline index_t NLLLossExp<Dtype>::dim(void) const {return 1;}

template<typename Dtype>
inline index_t NLLLossExp<Dtype>::size(index_t idx) const {return index_->size(0);}

template<typename Dtype>
inline bool NLLLossExp<Dtype>::requires_grad(void) const {return src_.requires_grad();}

template<typename Dtype>
inline Dtype NLLLossExp<Dtype>::eval(index_t* ids) const {
	index_t src_ids[2] = {*ids, index_->eval(ids)};
	return -src_->eval(src_ids);
}

template<typename Dtype>
inline void NLLLossExp<Dtype>::backward(const Exp<Dtype>& grad) const {
	GradExp src_grad(grad, *src_, *index_);
	ConstExptr<Dtype>::make_uncontrol(src_grad);
	src_.backward(src_grad);
}

// NLLLossExp
template<typename Dtype>
NLLLossExp<Dtype>::GradExp::GradExp(const Exp<Dtype>& grad, const Exp<Dtype>& src, const Exp<int_t>& index) {
	ConstExptr<Dtype>::make_uncontrol(grad);
	ConstExptr<Dtype>::make_uncontrol(src);
	ConstExptr<int_t>::make_uncontrol(index);
	grad_.reset(&grad, false);
	src_.reset(&src, false);
	index_.reset(&index, false);
}

template<typename Dtype>
index_t NLLLossExp<Dtype>::GradExp::dim(void) const {return 2;}

template<typename Dtype>
index_t NLLLossExp<Dtype>::GradExp::size(index_t idx) const {return src_->size(idx);}

template<typename Dtype>
bool NLLLossExp<Dtype>::GradExp::requires_grad(void) const {return src_->requires_grad();}

template<typename Dtype>
Dtype NLLLossExp<Dtype>::GradExp::eval(index_t *ids) const {
	index_t index_ids = ids[0];
	if(ids[1] == index_->eval(&index_ids))
		return -grad_->eval(&index_ids);
	else
		return 0;
}

template<typename Dtype>
void NLLLossExp<Dtype>::GradExp::backward(const Exp<Dtype>& grad) const {
	THROW_ERROR(NotImplementError, "Not Implement backward for NLL loss's gradient backward helper function.");
}

}  // namespace op
}  // namespace el



#endif