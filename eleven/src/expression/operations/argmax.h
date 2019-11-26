#ifndef EXPRESSION_OPERATIONS_ARGMAX_H_
#define EXPRESSION_OPERATIONS_ARGMAX_H_

#include "../expression.h"

namespace el {
namespace op {

template<typename Dtype>
struct ArgmaxExp: public UnaryExp<Dtype> {
public:	
	explicit ArgmaxExp(const Exp<Dtype>& operand, index_t dim);
	explicit ArgmaxExp(const Exp<Dtype>* operand, index_t dim);
	index_t dim(void) const;
	index_t size(index_t idx) const;
	Dtype eval(index_t* ids) const;
	void backward(const Exp<Dtype>& grad) const;
private:
	index_t dim_;
};

template<typename Dtype>
ArgmaxExp<Dtype>::ArgmaxExp(const Exp<Dtype>& operand, index_t dim)
	: UnaryExp<Dtype>(operand), dim_(dim) {}

template<typename Dtype>
ArgmaxExp<Dtype>::ArgmaxExp(const Exp<Dtype>* operand, index_t dim)
	: UnaryExp<Dtype>(operand), dim_(dim) {}

template<typename Dtype>
inline index_t ArgmaxExp<Dtype>::dim(void) const {return std::max(this->operand_->dim() - 1, 1);}

template<typename Dtype>
inline index_t ArgmaxExp<Dtype>::size(index_t idx) const {
	if(this->operand_->dim() == 1) return 1;
	if(idx < dim_) return this->operand_->size(idx);
	return this->operand_->size(idx + 1);
}

template<typename Dtype>
Dtype ArgmaxExp<Dtype>::eval(index_t* ids) const {
	index_t i;
	index_t src_dim = this->operand_->dim();
	index_t src_size = this->operand_->size(dim_);
	index_t* src_ids = new index_t[src_dim]();
	for(i = 0; i != dim_; i++)
		src_ids[i] = ids[i];
	for(i ++; i < src_dim; i++)
		src_ids[i] = ids[i-1];

	src_ids[dim_] = 0;
	index_t max_index = 0;
	Dtype value, max_value = this->operand_->eval(src_ids);
	for(i = 1; i < src_size; i++) {
		src_ids[dim_] = i;
		value = this->operand_->eval(src_ids);	
		if(value > max_value) {
			max_value = value;
			max_index = i;
		}
	}
	return max_index;
}	

template<typename Dtype>
void ArgmaxExp<Dtype>::backward(const Exp<Dtype>& grad) const {
	THROW_ERROR(NotImplementError, "not implement backward() for argmax operation.");
}

}  // namespace op
}  // namespace el

#endif