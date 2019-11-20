 #include "cross_entropy.h"

namespace el {
namespace nn {


Node<float_t> CrossEntropy::forward(const Node<float_t>& inputs, const Node<int_t>& labels) {
	auto log_softmax_node = op::log_softmax(inputs);
	auto nll_node = op::nll_loss(log_softmax_node, labels);
	auto reduce_loss = op::mean(nll_node, 0);
	Tensor<float_t>* result = new Tensor<float_t>(Shape(reduce_loss.get_exp()), true);
	*result = reduce_loss;
	return Node<float_t>(result);
}

}  // namespace nn
}  // namespace el