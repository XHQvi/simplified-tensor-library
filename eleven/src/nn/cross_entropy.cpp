 #include "cross_entropy.h"

namespace el {
namespace nn {


Node<float_t> CrossEntropy::forward(const Node<float_t>& inputs, const Node<int_t>& labels) {
	auto log_softmax_node = op::log_softmax(inputs);
	auto nll_node = op::nll_loss(log_softmax_node, labels);
	Tensor<float_t>* result = new Tensor<float_t>(Shape(nll_node.get_exp()), true);
	*result = nll_node;
	return Node<float_t>(result);
}

}  // namespace nn
}  // namespace el