#ifndef EXPRESSION_OPERATIONS_SUM_REDUCE_H_
#define EXPRESSION_OPERATIONS_SUM_REDUCE_H_

#include "../expression.h"

namespace el {
namespace op {

template<typename Dtype>
struct MeanReduceExp: public UnaryExp<Dtype> {
public:	
	explicit MeanReduceExp(const Exp<Dtype>& operand, index_t dim);
	explicit MeanReduceExp(const Exp<Dtype>* operand, index_t dim);
	index_t dim(void) const;
	index_t size(index_t idx) const;
	Dtype eval(index_t* ids) const;
	void backward(const Exp<Dtype>& grad) const;
private:
	index_t dim_;

	struct GradExp: public BinaryExp<Dtype> {
	public:
		explicit GradExp(const Exp<Dtype>& operand, const Exp<Dtype>& grad, index_t dim);
		index_t dim(void) const;
		index_t size(index_t idx) const;
		Dtype eval(index_t* ids) const;
		void backward(const Exp<Dtype>& grad) const;
	private:
		index_t dim_;
	};
};

template<typename Dtype>
MeanReduceExp<Dtype>::MeanReduceExp(const Exp<Dtype>& operand, index_t dim)
	: UnaryExp<Dtype>(operand), dim_(dim) {}

template<typename Dtype>
MeanReduceExp<Dtype>::MeanReduceExp(const Exp<Dtype>* operand, index_t dim)
	: UnaryExp<Dtype>(operand), dim_(dim) {}

template<typename Dtype>
inline index_t MeanReduceExp<Dtype>::dim(void) const {return std::max(this->operand_->dim() - 1, 1);}

template<typename Dtype>
inline index_t MeanReduceExp<Dtype>::size(index_t idx) const {
	if(this->operand_->dim() == 1) return 1;
	if(idx < dim_) return this->operand_->size(idx);
	return this->operand_->size(idx + 1);
}

template<typename Dtype>
Dtype MeanReduceExp<Dtype>::eval(index_t* ids) const {
	index_t i;
	index_t src_dim = this->operand_->dim();
	index_t src_size = this->operand_->size(dim_);
	index_t* src_ids = new index_t[src_dim];
	for(i = 0; i != dim_; i++)
		src_ids[i] = ids[i];
	for(i ++; i < src_dim; i++)
		src_ids[i] = ids[i-1];
	Dtype value = 0;
	for(i = 0; i < src_size; i++) {
		src_ids[dim_] = i;
		value += this->operand_->eval(src_ids);
	}
	return value / src_size;
}	

template<typename Dtype>
void MeanReduceExp<Dtype>::backward(const Exp<Dtype>& grad) const {
	GradExp sum_grad(*this->operand_, grad, dim_);
	ConstExptr<Dtype>::make_uncontrol(sum_grad);
	this->operand_.backward(sum_grad);
}

template<typename Dtype>
MeanReduceExp<Dtype>::GradExp::GradExp(const Exp<Dtype>& operand, 
									  const Exp<Dtype>& grad, 
									  index_t dim) 
	: BinaryExp<Dtype>(operand, grad), dim_(dim) {}

template<typename Dtype>
index_t MeanReduceExp<Dtype>::GradExp::dim(void) const {return this->loperand_->dim();}

template<typename Dtype>
index_t MeanReduceExp<Dtype>::GradExp::size(index_t idx) const {return this->loperand_->size(idx);}

template<typename Dtype>
Dtype MeanReduceExp<Dtype>::GradExp::eval(index_t *ids) const {
	index_t grad_dim = this->roperand_->dim();
	index_t *grad_ids = new index_t[grad_dim];
	index_t i;
	for(i = 0; i != dim_; i++)
		grad_ids[i] = ids[i];
	for(; i < grad_dim; i++)
		grad_ids[i] = ids[i+1];
	return this->roperand_->eval(grad_ids) / this->loperand_->size(dim_);
}

template<typename Dtype>
void MeanReduceExp<Dtype>::GradExp::backward(const Exp<Dtype>& grad) const {
	THROW_ERROR(NotImplementError, "Not implement backward() for sum reduce's gradient backward helper.");
}


}  // namespace op
}  // namespace el





#endif
