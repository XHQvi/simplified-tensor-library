#include "max_pooling.h"

namespace el {
namespace nn {

MaxPool2D::MaxPool2D(const std::pair<index_t, index_t>& kernel_size) 
	: kernel_size_(kernel_size) {}

Node<float_t> MaxPool2D::forward(const Node<float_t>& inputs) {
	auto pooling = op::maxpooling2d(inputs, kernel_size_);
	Tensor<float_t>* result = new Tensor<float_t>(Shape(pooling.get_exp()), true);
	*result = pooling;
	return Node<float_t>(result);
}

}  // namespace nn
}  // namespace el