#ifndef EXPRESSION_OPERATIONS_SIGMOID_H_
#define EXPRESSION_OPERATIONS_SIGMOID_H_

#include "base_ops.h"
#include "../expression.h"

namespace el {
namespace op {

template<typename Dtype>
struct SigmoidExp: public UnaryExp<Dtype> {
	explicit SigmoidExp(const Exp<Dtype>& operand): UnaryExp<Dtype>(operand) {}
	explicit SigmoidExp(const Exp<Dtype>* operand): UnaryExp<Dtype>(operand) {}
	Dtype eval(index_t* ids) const {return 1 / (1+std::exp(-this->operand_->eval(ids)));}
	
	struct GradExp: public UnaryExp<Dtype> {
		explicit GradExp(const Exp<Dtype>& operand): UnaryExp<Dtype>(operand) {}
		Dtype eval(index_t* ids) const {
			Dtype value = this->operand_->eval(ids);
			return value * (1 - value);
		}
		void backward(const Exp<Dtype>& grad) const {
			THROW_ERROR(NotImplementError, "Not Implement backward for sigmoid's grad  helper.");
		}
	};

	void backward(const Exp<Dtype>& grad) const {
		GradExp sigmoid_grad(*this);
		MulExp<Dtype> fgrad(sigmoid_grad, grad);
		ConstExptr<Dtype>::make_uncontrol(sigmoid_grad);
		ConstExptr<Dtype>::make_uncontrol(fgrad);
		this->operand_.backward(fgrad);
	}
};

}  // namespace op
}  // namespace el

#endif
