#ifndef EXPRESSION_OPERATIONS_MAX_POOLING_H_
#define EXPRESSION_OPERATIONS_MAX_POOLING_H_

#include "../expression.h"

namespace el {
namespace op {

template<typename Dtype>
struct MaxPool2DExp: public UnaryExp<Dtype> {
	explicit MaxPool2DExp(const Exp<Dtype>& operand,
						  const std::pair<index_t, index_t>& kernel_size);
	explicit MaxPool2DExp(const Exp<Dtype>* operand,
						  const std::pair<index_t, index_t>& kernel_size);
	index_t dim(void) const;
	index_t size(index_t idx) const;
	Dtype eval(index_t *ids) const;
	void backward(const Exp<Dtype>& grad) const;
private:
	std::pair<index_t, index_t> kernel_size_;
	std::pair<index_t, index_t> out_size_;

	struct GradExp: public BinaryExp<Dtype> {
	public:
		explicit GradExp(const Exp<Dtype>& operand,
						 const Exp<Dtype>& grad,
						 const std::pair<index_t, index_t>& kernel_size,
						 const std::pair<index_t, index_t>& out_size);
		index_t dim(void) const;
		index_t size(index_t idx) const;
		Dtype eval(index_t *ids) const;
		void backward(const Exp<Dtype>& grad) const;
	private:
		std::pair<index_t, index_t> kernel_size_;
		std::pair<index_t, index_t> out_size_;
	};
};

template<typename Dtype>
MaxPool2DExp<Dtype>::MaxPool2DExp(const Exp<Dtype>& operand,
								  const std::pair<index_t, index_t>& kernel_size) 
	: UnaryExp<Dtype>(operand), 
	  kernel_size_(kernel_size) {
	out_size_.first = 
		(operand.size(2) - kernel_size.first) / kernel_size.first + 1;
	out_size_.second = 
		(operand.size(3) - kernel_size.second) / kernel_size.second + 1;
}

template<typename Dtype>
MaxPool2DExp<Dtype>::MaxPool2DExp(const Exp<Dtype>* operand,
								  const std::pair<index_t, index_t>& kernel_size) 
	: UnaryExp<Dtype>(operand), 
	  kernel_size_(kernel_size) {
	out_size_.first = 
		(operand->size(2) - kernel_size.first) / kernel_size.first + 1;
	out_size_.second = 
		(operand->size(3) - kernel_size.second) / kernel_size.second + 1;
}

template<typename Dtype>
index_t MaxPool2DExp<Dtype>::dim(void) const {return 4;}

template<typename Dtype>
index_t MaxPool2DExp<Dtype>::size(index_t idx) const {
	if(idx < 2) return this->operand_->size(idx);
	return idx == 2 ? out_size_.first : out_size_.second;
}

template<typename Dtype>
Dtype MaxPool2DExp<Dtype>::eval(index_t* ids) const {
	index_t h_start = ids[2] * kernel_size_.first;
	index_t h_end   = h_start + kernel_size_.first;
	index_t w_start = ids[3] * kernel_size_.second;
	index_t w_end   = w_start + kernel_size_.second;
	index_t loc[4] = {ids[0], ids[1], h_start, w_start};  // batch index and channel index

	Dtype value = this->operand_->eval(loc);
	for(index_t i = h_start; i < h_end; i++) {
		loc[2] = i;
		for(index_t j = w_start; j < w_end; j++) {
			loc[3] = j;
			value = std::max(value, this->operand_->eval(loc));
		}
	}
	return value;
}

template<typename Dtype>
void MaxPool2DExp<Dtype>::backward(const Exp<Dtype>& grad) const {
	GradExp maxpooling_grad(*this->operand_, grad, kernel_size_, out_size_);
	ConstExptr<Dtype>::make_uncontrol(maxpooling_grad);
	this->operand_.backward(maxpooling_grad);
}

template<typename Dtype>
MaxPool2DExp<Dtype>::GradExp::GradExp(const Exp<Dtype>& operand,
									  const Exp<Dtype>& grad,
									  const std::pair<index_t, index_t>& kernel_size,
									  const std::pair<index_t, index_t>& out_size)
	: BinaryExp<Dtype>(operand, grad), 
	  kernel_size_(kernel_size),
	  out_size_(out_size) {}

template<typename Dtype>
index_t MaxPool2DExp<Dtype>::GradExp::dim(void) const {return 4;}

template<typename Dtype>
index_t MaxPool2DExp<Dtype>::GradExp::size(index_t idx) const {
	return this->loperand_->size(idx);
}

template<typename Dtype>
Dtype MaxPool2DExp<Dtype>::GradExp::eval(index_t *ids) const {
	index_t h_start = ids[2] / kernel_size_.first * kernel_size_.first;
	index_t h_end   = h_start + kernel_size_.first;
	index_t w_start = ids[3] / kernel_size_.second * kernel_size_.second;
	index_t w_end   = w_start + kernel_size_.second;
	index_t loc[4] = {ids[0], ids[1], h_start, w_start};  // batch index and channel index

	Dtype value = this->loperand_->eval(ids);
	for(index_t i = h_start; i < h_end; i++) {
		loc[2] = i;
		for(index_t j = w_start; j < w_end; j++) {
			loc[3] = j;
			if(this->loperand_->eval(loc) > value)
				return 0;			
		}
	}
	loc[2] = h_start / kernel_size_.first;
	loc[3] = w_start / kernel_size_.second;
	return this->roperand_->eval(loc);
}

template<typename Dtype>
void MaxPool2DExp<Dtype>::GradExp::backward(const Exp<Dtype>& grad) const {
	THROW_ERROR(NotImplementError, 
		"Not Implement backward for max poolingg's gradient backward helper.");
}

}  // namespace op
}  // namespace el


#endif