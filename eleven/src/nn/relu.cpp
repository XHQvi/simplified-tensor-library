#include "relu.h"


namespace el {
namespace nn {

Node<float_t> ReLU::forward(const Node<float_t>& inputs) {
	auto relu = op::relu(inputs);
	Tensor<float_t>* result = new Tensor<float_t>(Shape(relu.get_exp()), true);
	*result = relu;
	return Node<float_t>(result);
}

}  // namespace nn
}  // namespace el