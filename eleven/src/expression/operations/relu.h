#ifndef EXPRESSION_OPERATIONS_RELU_H_
#define EXPRESSION_OPERATIONS_RELU_H_

#include "../expression.h"

namespace el {
namespace op {

template<typename Dtype>
struct ReLUExp: public UnaryExp<Dtype> {
	explicit ReLUExp(const Exp<Dtype>& operand): UnaryExp<Dtype>(operand) {}
	explicit ReLUExp(const Exp<Dtype>* operand): UnaryExp<Dtype>(operand) {}
	Dtype eval(index_t* ids) const {return std::max((Dtype)0, this->operand_->eval(ids));}
	
	struct GradExp: public BinaryExp<Dtype> {
		explicit GradExp(const Exp<Dtype>& loperand, const Exp<Dtype>& roperand)
			: BinaryExp<Dtype>(loperand, roperand) {}
		Dtype eval(index_t* ids) const {
			return this->loperand_->eval(ids) > 0 ? this->roperand_->eval(ids) : 0;
		}
		void backward(const Exp<Dtype>& grad) const {
			THROW_ERROR(NotImplementError, "Not Implement backward for relu's grad  helper.");
		}
	};

	void backward(const Exp<Dtype>& grad) const {
		GradExp relu_grad(*this->operand_, grad);
		ConstExptr<Dtype>::make_uncontrol(relu_grad);
		this->operand_.backward(relu_grad);
	}
};


}  // namespace el
}  // namespace op

#endif
