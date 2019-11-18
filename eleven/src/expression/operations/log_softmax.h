#ifndef EXPRESSION_OPERATIONS_LOG_SOFTMAX_H_
#define EXPRESSION_OPERATIONS_LOG_SOFTMAX_H_

#include <cmath>
#include "../expression.h"

namespace el {
namespace op {

template<typename Dtype>
struct LogSoftmaxExp: public UnaryExp<Dtype> {
	explicit LogSoftmaxExp(const Exp<Dtype>& operand);
	explicit LogSoftmaxExp(const Exp<Dtype>* operand);
	Dtype eval(index_t* ids) const;
	void backward(const Exp<Dtype>& grad) const;

	struct GradExp: public BinaryExp<Dtype> {
		explicit GradExp(const Exp<Dtype>& operand, const Exp<Dtype>& grad, Dtype exp_sum, Dtype max_item);
		Dtype eval(index_t* ids) const;
		void backward(const Exp<Dtype>& grad) const;
	private:
		Dtype exp_sum_;
		Dtype max_item_;
	};

private:
	Dtype exp_sum_;
	Dtype log_exp_sum_;
	Dtype max_item_;
};

template<typename Dtype>
LogSoftmaxExp<Dtype>::LogSoftmaxExp(const Exp<Dtype>& operand)
	: UnaryExp<Dtype>(operand), exp_sum_(0) {
	
	index_t i = 0;
	max_item_ = operand.eval(&i);
	for(i++; i < operand.size(0); i ++) 
		max_item_ = std::max(operand.eval(&i), max_item_);
	for(i = 0; i < operand.size(0); i++)
		exp_sum_ += std::exp(operand.eval(&i) - max_item_);
	log_exp_sum_ = std::log(exp_sum_);
}

template<typename Dtype>
LogSoftmaxExp<Dtype>::LogSoftmaxExp(const Exp<Dtype>* operand)
	: UnaryExp<Dtype>(operand), exp_sum_(0) {
	
	index_t i = 0;
	max_item_ = operand->eval(&i);
	for(i++; i < operand->size(0); i ++) 
		max_item_ = std::max(operand->eval(&i), max_item_);
	for(i = 0; i < operand->size(0); i++)
		exp_sum_ += std::exp(operand->eval(&i) - max_item_);
	log_exp_sum_ = std::log(exp_sum_);
}

template<typename Dtype>
inline Dtype LogSoftmaxExp<Dtype>::eval(index_t* ids) const {
	return this->operand_->eval(ids) - max_item_ - log_exp_sum_;
}

template<typename Dtype>
inline void LogSoftmaxExp<Dtype>::backward(const Exp<Dtype>& grad) const {
	GradExp logsoftmax_grad(*this->operand_, grad, exp_sum_, max_item_);
	ConstExptr<Dtype>::make_uncontrol(logsoftmax_grad);
	this->operand_.backward(logsoftmax_grad);
}

template<typename Dtype>
LogSoftmaxExp<Dtype>::GradExp::GradExp(const Exp<Dtype>& operand, const Exp<Dtype>& grad, Dtype exp_sum, Dtype max_item) 
	: BinaryExp<Dtype>(operand, grad), exp_sum_(exp_sum), max_item_(max_item) {}

template<typename Dtype>
inline Dtype LogSoftmaxExp<Dtype>::GradExp::eval(index_t* ids) const {
	Dtype grad = (1 - this->loperand_->size(0) * std::exp(this->loperand_->eval(ids) - max_item_) / exp_sum_);
	return this->roperand_->eval(ids) * grad;
}

template<typename Dtype>
inline void LogSoftmaxExp<Dtype>::GradExp::backward(const Exp<Dtype>& grad) const {
	THROW_ERROR(NotImplementError, "Not Implement backward for log softmax's gradient backward helper.");
}

}  // namespace op
}  // namespace el
#endif
