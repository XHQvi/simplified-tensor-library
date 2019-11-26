#ifndef EXPRESSION_OPERATIONS_MATRIX_MULTIPLY_H_
#define EXPRESSION_OPERATIONS_MATRIX_MULTIPLY_H_

#include "base_ops.h"


namespace el {
namespace op {

template<typename Dtype>
struct MMExp: public BinaryExp<Dtype> {
	MMExp(const Exp<Dtype>& loperand, const Exp<Dtype>& roperand): BinaryExp<Dtype>(loperand, roperand){}
	MMExp(const Exp<Dtype>* loperand, const Exp<Dtype>* roperand): BinaryExp<Dtype>(loperand, roperand){}
	index_t dim(void) const {return 2;}
	index_t size(index_t idx) const {return idx == 0 ? this->loperand_->size(0) : this->roperand_->size(1);}
	Dtype eval(index_t* ids) const {
		Dtype value = 0;
		index_t l_loc[2] = {ids[0], 0};
		index_t r_loc[2] = {0, ids[1]};
		for(index_t i = 0; i < this->loperand_->size(1); i++) {
			l_loc[1] = i;
			r_loc[0] = i;
			value += this->loperand_->eval(l_loc) * this->roperand_->eval(r_loc);
		}
		return value;
	}
	void backward(const Exp<Dtype>& grad) const {
		MatrixTransposeExp<Dtype> rtranspose(*this->roperand_);
		MMExp<Dtype> lgrad(grad, rtranspose);
		ConstExptr<Dtype>::make_uncontrol(rtranspose);
		ConstExptr<Dtype>::make_uncontrol(lgrad);
		this->loperand_.backward(lgrad);

		MatrixTransposeExp<Dtype> ltranspose(*this->loperand_);
		MMExp<Dtype> rgrad(ltranspose, grad);
		ConstExptr<Dtype>::make_uncontrol(ltranspose);
		ConstExptr<Dtype>::make_uncontrol(rgrad);
		this->roperand_.backward(rgrad);
	} 
};

template<typename Dtype>
struct BMMExp: public BinaryExp<Dtype> {
	BMMExp(const Exp<Dtype>& loperand, const Exp<Dtype>& roperand)
		: BinaryExp<Dtype>(loperand, roperand) {}
	BMMExp(const Exp<Dtype>* loperand, const Exp<Dtype>* roperand)
		: BinaryExp<Dtype>(loperand, roperand) {}
	index_t dim(void) const {return 3;}
	index_t size(index_t idx) const {
		switch(idx) {
			case 0: return std::max(this->roperand_->size(0), this->loperand_->size(0));
			case 1: return this->loperand_->size(1);
			default: return this->roperand_->size(2);
		}
	}
	Dtype eval(index_t* ids) const {
		Dtype value = 0;
		index_t l_loc[3] = {ids[0], ids[1], 0};
		index_t r_loc[3] = {ids[0], 0, ids[2]};
		for(index_t i = 0; i < this->loperand_->size(2); i++) {
			l_loc[2] = i;
			r_loc[1] = i;
			value += this->loperand_->eval(l_loc) * this->roperand_->eval(r_loc);
		}
		return value;
	}
	struct BMTExp: public UnaryExp<Dtype> {
		BMTExp(const Exp<Dtype>& operand): UnaryExp<Dtype>(operand) {}
		index_t dim(void) const {return 3;}
		index_t size(index_t idx) const {
			switch(idx) {
				case 0: return this->operand_->size(0);
				case 1: return this->operand_->size(2);
				default: return this->operand_->size(1);
			}
		}
		Dtype eval(index_t* ids) const {
			index_t trans_ids[3] = {ids[0], ids[2], ids[1]};
			return this->operand_->eval(trans_ids);
		}
		void backward(const Exp<Dtype>& grad) const {
			THROW_ERROR(NotImplementError, "Not Implement backward for Batch Matrix Transpose.");
		}
	};
	void backward(const Exp<Dtype>& grad) const {
		BMTExp rbmt(*this->roperand_);
		BMMExp<Dtype> lgrad(grad, rbmt);
		ConstExptr<Dtype>::make_uncontrol(rbmt);
		ConstExptr<Dtype>::make_uncontrol(lgrad);
		this->loperand_.backward(lgrad);

		BMTExp lbmt(*this->loperand_);
		BMMExp<Dtype> rgrad(lbmt, grad);
		ConstExptr<Dtype>::make_uncontrol(lbmt);
		ConstExptr<Dtype>::make_uncontrol(rgrad);
		this->roperand_.backward(rgrad);
	}
};

}  // namespace op
}  // namespace el

#endif
