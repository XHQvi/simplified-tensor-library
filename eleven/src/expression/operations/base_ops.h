#ifndef EXPRESSION_OPERATIONS_BASE_OPS_H_
#define EXPRESSION_OPERATIONS_BASE_OPS_H_

#include "../expression.h"

namespace el {
namespace op {

template<typename Dtype>
struct MinusExp: public UnaryExp<Dtype> {
	explicit MinusExp(const Exp<Dtype>& operand): UnaryExp<Dtype>(operand) {}
	explicit MinusExp(const Exp<Dtype>* operand): UnaryExp<Dtype>(operand) {}
	Dtype eval(index_t* ids) const {return -this->operand_->eval(ids);}
	void backward(const Exp<Dtype>& grad) const {
		MinusExp<Dtype> minus_grad(grad);
		ConstExptr<Dtype>::make_uncontrol(minus_grad);
		this->operand_.backward(minus_grad);
	}
};

template<typename Dtype>
struct MatrixTransposeExp: public UnaryExp<Dtype> {
	explicit MatrixTransposeExp(const Exp<Dtype>& operand): UnaryExp<Dtype>(operand) {}
	explicit MatrixTransposeExp(const Exp<Dtype>* operand): UnaryExp<Dtype>(operand) {}
	index_t dim(void) const {return 2;}
	index_t size(index_t idx) const {return idx == 0 ? this->operand_->size(1) : this->operand_->size(0);}
	Dtype eval(index_t* ids) const {
		index_t trans_ids[2] = {ids[1], ids[0]};
		return this->operand_->eval(trans_ids);
	}
	void backward(const Exp<Dtype>& grad) const {
		MatrixTransposeExp<Dtype> trans_grad(grad);
		ConstExptr<Dtype>::make_uncontrol(trans_grad);
		this->operand_.backward(trans_grad);
	}
};

template<typename Dtype>
struct AddExp: public BinaryExp<Dtype> {
	AddExp(const Exp<Dtype>& loperand, const Exp<Dtype>& roperand): BinaryExp<Dtype>(loperand, roperand){}
	AddExp(const Exp<Dtype>* loperand, const Exp<Dtype>* roperand): BinaryExp<Dtype>(loperand, roperand) {}
	Dtype eval(index_t* ids) const {return this->loperand_->eval(ids) + this->roperand_->eval(ids);}
	void backward(const Exp<Dtype>& grad) const {
		this->loperand_.backward(grad);
		this->roperand_.backward(grad);
	}
};

template<typename Dtype>
struct SubExp: public BinaryExp<Dtype> {
	SubExp(const Exp<Dtype>& loperand, const Exp<Dtype>& roperand): BinaryExp<Dtype>(loperand, roperand){}
	SubExp(const Exp<Dtype>* loperand, const Exp<Dtype>* roperand): BinaryExp<Dtype>(loperand, roperand){}
	Dtype eval(index_t* ids) const {return this->loperand_->eval(ids) - this->roperand_->eval(ids);}
	void backward(const Exp<Dtype>& grad) const {
		this->loperand_.backward(grad);

		MinusExp<Dtype> minus_grad(grad);
		ConstExptr<Dtype>::make_uncontrol(minus_grad);
		this->roperand_.backward(minus_grad);
	}
};

template<typename Dtype>
struct MulExp: public BinaryExp<Dtype> {
	MulExp(const Exp<Dtype>& loperand, const Exp<Dtype>& roperand): BinaryExp<Dtype>(loperand, roperand){}
	MulExp(const Exp<Dtype>* loperand, const Exp<Dtype>* roperand): BinaryExp<Dtype>(loperand, roperand){}
	Dtype eval(index_t* ids) const {return this->loperand_->eval(ids) * this->roperand_->eval(ids);}
	void backward(const Exp<Dtype>& grad) const {
		MulExp<Dtype> lgrad(grad, *this->roperand_);
		ConstExptr<Dtype>::make_uncontrol(lgrad);
		this->loperand_.backward(lgrad);
		MulExp<Dtype> rgrad(grad, *this->loperand_);
		ConstExptr<Dtype>::make_uncontrol(rgrad);
		this->roperand_.backward(rgrad);
	}
};

}  // namespace op
}  // namespace el

#endif
