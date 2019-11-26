#ifndef EXPRESSION_OPERATIONS_IMG2COL_H_
#define EXPRESSION_OPERATIONS_IMG2COL_H_

#include "../expression.h"

namespace el {
namespace op {

template<typename Dtype>
struct Img2ColExp: public UnaryExp<Dtype> {
	explicit Img2ColExp(const Exp<Dtype>& operand, 
						const std::pair<index_t, index_t>& kernel_size, 
					    const std::pair<index_t, index_t>& stride, 
					    const std::pair<index_t, index_t>& padding);

	explicit Img2ColExp(const Exp<Dtype>* operand, 
						const std::pair<index_t, index_t>& kernel_size, 
					    const std::pair<index_t, index_t>& stride, 
					    const std::pair<index_t, index_t>& padding);

	index_t dim(void) const;
	index_t out_size(index_t idx) const;
	index_t size(index_t idx) const;
	Dtype eval(index_t* ids) const;
	void backward(const Exp<Dtype>& grad) const;

private:
	std::pair<index_t, index_t> kernel_size_;
	std::pair<index_t, index_t> stride_;
	std::pair<index_t, index_t> padding_;
	std::pair<index_t, index_t> out_size_;  // feature map's size after conv

	struct GradExp: public BinaryExp<Dtype> {
		explicit GradExp(const Exp<Dtype>& operand,
						 const Exp<Dtype>& grad,
						 const std::pair<index_t, index_t>& kernel_size, 
					     const std::pair<index_t, index_t>& stride, 
					     const std::pair<index_t, index_t>& padding,
					     const std::pair<index_t, index_t>& out_size);
		index_t dim(void) const;
		index_t size(index_t idx) const;
		Dtype eval(index_t* ids) const;
		void backward(const Exp<Dtype>& grad) const;

		std::pair<index_t, index_t> kernel_size_;
		std::pair<index_t, index_t> stride_;
		std::pair<index_t, index_t> padding_;
		std::pair<index_t, index_t> img_size_;
		std::pair<index_t, index_t> out_size_;
	};
};

template<typename Dtype>
Img2ColExp<Dtype>::Img2ColExp(const Exp<Dtype>& operand, 
					          const std::pair<index_t, index_t>& kernel_size, 
				              const std::pair<index_t, index_t>& stride, 
				              const std::pair<index_t, index_t>& padding)
	: UnaryExp<Dtype>(operand), kernel_size_(kernel_size), stride_(stride), padding_(padding) {
	out_size_.first = 
		(this->operand_->size(2) + 2 * padding_.first - kernel_size_.first) / stride_.first + 1;
	out_size_.second = 
		(this->operand_->size(3) + 2 * padding_.second - kernel_size_.second) / stride_.second + 1;
}

template<typename Dtype>
Img2ColExp<Dtype>::Img2ColExp(const Exp<Dtype>* operand, 
					          const std::pair<index_t, index_t>& kernel_size, 
				              const std::pair<index_t, index_t>& stride, 
				              const std::pair<index_t, index_t>& padding)
	: UnaryExp<Dtype>(operand), kernel_size_(kernel_size), stride_(stride), padding_(padding) {
	out_size_.first = 
		(this->operand_->size(2) + 2 * padding_.first - kernel_size_.first) / stride_.first + 1;
	out_size_.second = 
		(this->operand_->size(3) + 2 * padding_.second - kernel_size_.second) / stride_.second + 1;
}

template<typename Dtype>
index_t Img2ColExp<Dtype>::dim(void) const {return 3;}

template<typename Dtype>
index_t Img2ColExp<Dtype>::out_size(index_t idx) const {
	return idx == 0 ? out_size_.first : out_size_.second;
}

// A batch of images, whose size is (b, c, h, w), will be packed into b matrixes with size of (c*kh*kw, oh*ow),
// where {kh, kw}, {oh, ow} is kernel size and size of feature map after conv.
// Then unpack the weight into a matrix with (oc, c*kh*kw), where oc is conv's output channels.
// Just bmm(weight_mats, image_mats) and get the result of conv.
//
// It seems that usual implement is packing the images into a matrix (c*kh*kw, oh*ow*b). So to get the result of 
// convolution, just use mm(weight_mat, image_mat) instead of bmm.
template<typename Dtype>
index_t Img2ColExp<Dtype>::size(index_t idx) const {
	switch(idx) {
		case 0: return this->operand_->size(0);  // batch size
		case 1: return this->operand_->size(1) * kernel_size_.first * kernel_size_.second;  // c*kh*kw
		default: return out_size_.first * out_size_.second;
	}
}

template<typename Dtype>
Dtype Img2ColExp<Dtype>::eval(index_t* ids) const {
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
	if(loc[2] < 0 || loc[2] >= this->operand_->size(2) || 
	   loc[3] < 0 || loc[3] >= this->operand_->size(3))
		return 0;
	return this->operand_->eval(loc);
}

template<typename Dtype>
void Img2ColExp<Dtype>::backward(const Exp<Dtype>& grad) const {
	GradExp img2col_grad(*this->operand_, grad,
						 kernel_size_, stride_, padding_, out_size_);
	ConstExptr<Dtype>::make_uncontrol(img2col_grad);
	this->operand_.backward(img2col_grad);
}

template<typename Dtype>
Img2ColExp<Dtype>::GradExp::GradExp(const Exp<Dtype>& operand,
									const Exp<Dtype>& grad,
									const std::pair<index_t, index_t>& kernel_size, 
								    const std::pair<index_t, index_t>& stride, 
					    			const std::pair<index_t, index_t>& padding,
					    			const std::pair<index_t, index_t>& out_size)
	: BinaryExp<Dtype>(operand, grad),
	  kernel_size_(kernel_size),
	  stride_(stride),
	  padding_(padding),
	  out_size_(out_size) {}

template<typename Dtype>
inline index_t Img2ColExp<Dtype>::GradExp::dim(void) const {
	return this->loperand_->dim();
}

template<typename Dtype>
inline index_t Img2ColExp<Dtype>::GradExp::size(index_t idx) const {
	return this->loperand_->size(idx);
}

template<typename Dtype>
Dtype Img2ColExp<Dtype>::GradExp::eval(index_t* ids) const {
	index_t img_h = size(2), img_w = size(3);
	index_t kh_idx, kw_idx;  // indice in a patch, ranging from 0 to kernel_size
	index_t ph_idx, pw_idx;  // indice of the patch, ranging from 0 to output_size
	index_t loc[3] = {ids[0], 0, 0};  // keep the batch index
	Dtype total_grad = 0;

	// iterate all possible (kh_idx, kw_idx)
	for(kh_idx = 0; kh_idx < kernel_size_.first; kh_idx ++) {
		for(kw_idx = 0; kw_idx < kernel_size_.second; kw_idx ++) {
			// calculate corresponding (ph, pw)
			ph_idx = ids[2] - kh_idx + padding_.first;
			pw_idx = ids[3] - kw_idx + padding_.second;

			// check (ph_idx, pw_idx) valid
			if(ph_idx < 0 || ph_idx + kernel_size_.first > img_h + padding_.first ||
			   pw_idx < 0 || pw_idx + kernel_size_.second > img_w + padding_.second)
				continue;
			if(ph_idx % stride_.first || pw_idx % stride_.second)
				continue;
			// transform (ph_idx, pw_idx) and (kh_idx, kw_idx) to indice for Img2Col's output,
			// and accumulate the gradient.
			loc[1] = ids[1] * kernel_size_.first * kernel_size_.second +
                     kh_idx * kernel_size_.second +
                     kw_idx;
            loc[2] = ph_idx / stride_.first * out_size_.second +
            		 pw_idx / stride_.second;
            total_grad += this->roperand_->eval(loc);
		}
	}

	return total_grad;
}

template<typename Dtype>
void Img2ColExp<Dtype>::GradExp::backward(const Exp<Dtype>& grad) const {
	THROW_ERROR(NotImplementError, 
		"Not implement backward for img2col's gradient backward.");
}


}  // namespace op
}  // namespace el

#endif