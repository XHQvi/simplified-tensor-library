#ifndef EXPRESSION_UNARY_EXP_H_
#define EXPRESSION_UNARY_EXP_H_

#include <cmath>
#include "expression.h"
#include <iostream>

namespace el {

template<typename Dtype> class Exp;
template<typename Dtype> class UnaryExp;
template<typename Dtype> class Tensor;

namespace op {

template<typename Dtype>
struct AbsExp: public UnaryExp<Dtype> {
	explicit AbsExp(const Exp<Dtype>& operand): UnaryExp<Dtype>(operand) {}
	explicit AbsExp(const Exp<Dtype>* operand): UnaryExp<Dtype>(operand) {}
	Dtype eval(index_t* ids) const {return std::abs(this->operand_->eval(ids));}
};

template<typename Dtype>
struct SigmoidExp: public UnaryExp<Dtype> {
	explicit SigmoidExp(const Exp<Dtype>& operand): UnaryExp<Dtype>(operand) {}
	explicit SigmoidExp(const Exp<Dtype>* operand): UnaryExp<Dtype>(operand) {}
	Dtype eval(index_t* ids) const {return 1 / (1+std::exp(-this->operand_->eval(ids)));}
};

template<typename Dtype>
struct TransposeExp: public UnaryExp<Dtype> {
	explicit TransposeExp(const Exp<Dtype>& operand): UnaryExp<Dtype>(operand) {}
	explicit TransposeExp(const Exp<Dtype>* operand): UnaryExp<Dtype>(operand) {}
	Dtype eval(index_t* ids) const {
		index_t trans_ids[2] = {ids[1], ids[0]};
		return this->operand_->eval(trans_ids);
	}
};

template<typename Dtype>
struct Img2ColExp: public UnaryExp<Dtype> {
	std::pair<index_t, index_t> kernel_size_;
	std::pair<index_t, index_t> stride_;
	std::pair<index_t, index_t> padding_;
	std::pair<index_t, index_t> out_size_;  // feature map's size after conv
	explicit Img2ColExp(const Exp<Dtype>& operand, 
						const std::pair<index_t, index_t> kernel_size, 
					    const std::pair<index_t, index_t> stride, 
					    const std::pair<index_t, index_t> padding)
		: UnaryExp<Dtype>(operand), kernel_size_(kernel_size), stride_(stride), padding_(padding) {
		out_size_.first = 
			(this->operand_->size(2) + 2 * padding_.first - kernel_size_.first) / stride_.first + 1;
		out_size_.second = 
			(this->operand_->size(3) + 2 * padding_.second - kernel_size_.second) / stride_.second + 1;
	}
	explicit Img2ColExp(const Exp<Dtype>* operand, 
						const std::pair<index_t, index_t> kernel_size, 
					    const std::pair<index_t, index_t> stride, 
					    const std::pair<index_t, index_t> padding)
		: UnaryExp<Dtype>(operand), kernel_size_(kernel_size), stride_(stride), padding_(padding) {
		out_size_.first = 
			(this->operand_->size(2) + 2 * padding_.first - kernel_size_.first) / stride_.first + 1;
		out_size_.second = 
			(this->operand_->size(3) + 2 * padding_.second - kernel_size_.second) / stride_.second + 1;
	}

	index_t dim(void) const {return 3;}
	index_t out_size(index_t idx) const {return idx == 0 ? out_size_.first : out_size_.second;}

	// A batch of images, whose size is (b, c, h, w), will be unpack into b matrixes with size of (c*kh*kw, oh*ow),
	// where {kh, kw}, {oh, ow} is kernel size and size of feature map after conv.
	// Then unpack the weight into a matrix with (oc, c*kh*kw), where oc is conv's output channels.
	// Just dot(weight_mat, image_mat) and get the result of conv.
	index_t size(index_t idx) const {
		switch(idx) {
			case 0: return this->operand_->size(0);  // batch size
			case 1: return this->operand_->size(1) * kernel_size_.first * kernel_size_.second;  // c*kh*kw
			default: return out_size_.first * out_size_.second;
		}
	}

	Dtype eval(index_t* ids) const {
		index_t loc[4];
		loc[0] = ids[0];  // batch index

		loc[1] = ids[1] / (kernel_size_.first * kernel_size_.second);  // channel index
		index_t kloc_idx = ids[1] % (kernel_size_.first * kernel_size_.second);
		index_t kh_idx = kloc_idx / kernel_size_.second;
		index_t kw_idx = kloc_idx % kernel_size_.second;

		index_t h_idx = ids[2] / out_size_.second;
		index_t w_idx = ids[2] % out_size_.second;
		h_idx = h_idx * stride_.first - padding_.first;
		w_idx = w_idx * stride_.second - padding_.second;

		// {h_idx, w_idx} can be seen as the location of a patch's first pixel(top left pixel),
		// {kh_idx, kw_idx} can be seen as a pixel's location in this patch.
		// The final location can be out of the origin img because of padding, and just return 0.
		loc[2] = h_idx + kh_idx;
		loc[3] = w_idx + kw_idx;
		if(loc[2] < 0 || loc[2] >= this->operand_->size(2) || loc[3] < 0 || loc[3] >= this->operand_->size(3))
			return 0;
		return this->operand_->eval(loc);
	}

	void backward(const Exp<Dtype>& grad) const {
		
	}
};

} // namespace op
} // namespace el

#endif