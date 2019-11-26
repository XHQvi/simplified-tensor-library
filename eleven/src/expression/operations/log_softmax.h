#ifndef EXPRESSION_OPERATIONS_LOG_SOFTMAX_H_
#define EXPRESSION_OPERATIONS_LOG_SOFTMAX_H_

#include <cmath>
#include <memory>
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
		explicit GradExp(const Exp<Dtype>& operand, 
			             const Exp<Dtype>& grad, 
			             const std::shared_ptr<Dtype>& exp_sum, 
			             const std::shared_ptr<Dtype>& max_item);
		Dtype eval(index_t* ids) const;
		void backward(const Exp<Dtype>& grad) const;
	private:
		const std::shared_ptr<Dtype> exp_sum_;
		const std::shared_ptr<Dtype> max_item_;
	};

private:
	std::shared_ptr<Dtype> exp_sum_;
	std::shared_ptr<Dtype> log_exp_sum_;
	std::shared_ptr<Dtype> max_item_;
};

template<typename Dtype>
LogSoftmaxExp<Dtype>::LogSoftmaxExp(const Exp<Dtype>& operand)
	: UnaryExp<Dtype>(operand),
	  exp_sum_(new Dtype[operand.size(0)]{0}, std::default_delete<Dtype[]>()),
	  log_exp_sum_(new Dtype[operand.size(0)], std::default_delete<Dtype[]>()),
	  max_item_(new Dtype[operand.size(0)], std::default_delete<Dtype[]>()) {
	
	auto exp_sum_ptr = exp_sum_.get();
	auto log_exp_sum_ptr = log_exp_sum_.get();
	auto max_item_ptr = max_item_.get();
	index_t ids[2];
	index_t num_batch = operand.size(0);
	index_t num_cls = operand.size(1);

	for(index_t i = 0; i < num_batch; i++) {
		ids[0] = i;
		ids[1] = 0;
		max_item_ptr[i] = operand.eval(ids);
		for(index_t j = 1; j < num_cls; j++) {
			ids[1] = j;
			max_item_ptr[i] = std::max(operand.eval(ids), max_item_ptr[i]);
		}
		for(index_t j = 0; j < num_cls; j++) {
			ids[1] = j;
			exp_sum_ptr[i] += std::exp(operand.eval(ids) - max_item_ptr[i]);
		}
		log_exp_sum_ptr[i] = std::log(exp_sum_ptr[i]);
	}
}

template<typename Dtype>
LogSoftmaxExp<Dtype>::LogSoftmaxExp(const Exp<Dtype>* operand)
	: UnaryExp<Dtype>(operand),
	  exp_sum_(new Dtype[operand->size(0)]{0}, std::default_delete<Dtype[]>()),
	  log_exp_sum_(new Dtype[operand->size(0)], std::default_delete<Dtype[]>()),
	  max_item_(new Dtype[operand->size(0)], std::default_delete<Dtype[]>()) {
	
	auto exp_sum_ptr = exp_sum_.get();
	auto log_exp_sum_ptr = log_exp_sum_.get();
	auto max_item_ptr = max_item_.get();
	index_t ids[2];
	index_t num_batch = operand->size(0);
	index_t num_cls = operand->size(1);

	for(index_t i = 0; i < num_batch; i++) {
		ids[0] = i;
		ids[1] = 0;
		max_item_ptr[i] = operand->eval(ids);
		for(index_t j = 1; j < num_cls; j++) {
			ids[1] = j;
			max_item_ptr[i] = std::max(operand->eval(ids), max_item_ptr[i]);
		}
		for(index_t j = 0; j < num_cls; j++) {
			ids[1] = j;
			exp_sum_ptr[i] += std::exp(operand->eval(ids) - max_item_ptr[i]);
		}
		log_exp_sum_ptr[i] = std::log(exp_sum_ptr[i]);
	}
}

template<typename Dtype>
inline Dtype LogSoftmaxExp<Dtype>::eval(index_t* ids) const {
	return this->operand_->eval(ids) - max_item_.get()[ids[0]] - log_exp_sum_.get()[ids[0]];
}

template<typename Dtype>
inline void LogSoftmaxExp<Dtype>::backward(const Exp<Dtype>& grad) const {
	GradExp logsoftmax_grad(*this->operand_, grad, exp_sum_, max_item_);
	ConstExptr<Dtype>::make_uncontrol(logsoftmax_grad);
	this->operand_.backward(logsoftmax_grad);
}

template<typename Dtype>
LogSoftmaxExp<Dtype>::GradExp::GradExp(const Exp<Dtype>& operand, 
			             			   const Exp<Dtype>& grad, 
			                           const std::shared_ptr<Dtype>& exp_sum, 
			                           const std::shared_ptr<Dtype>& max_item) 
	: BinaryExp<Dtype>(operand, grad), exp_sum_(exp_sum), max_item_(max_item) {}

template<typename Dtype>
inline Dtype LogSoftmaxExp<Dtype>::GradExp::eval(index_t* ids) const {
	index_t num_cls = this->loperand_->size(1);
	index_t grad_ids[2] = {ids[0], 0};
	Dtype total_grad = 0;
	Dtype softmax = std::exp(this->loperand_->eval(ids) - max_item_.get()[ids[0]]) / exp_sum_.get()[ids[0]];

	for(index_t i = 0; i < num_cls; i++) {
		grad_ids[1] = i;
		if(ids[1] != i) {
			total_grad -= softmax * this->roperand_->eval(grad_ids);
		} else { 
			total_grad += (1-softmax) * this->roperand_->eval(grad_ids);
		}
	}
	return total_grad;
}

template<typename Dtype>
inline void LogSoftmaxExp<Dtype>::GradExp::backward(const Exp<Dtype>& grad) const {
	THROW_ERROR(NotImplementError, "Not Implement backward for log softmax's gradient backward helper.");
}

}  // namespace op
}  // namespace el
#endif
