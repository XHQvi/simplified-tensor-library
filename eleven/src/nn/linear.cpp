#include "linear.h"

namespace el {
namespace nn {

Linear::Linear(index_t in_features, index_t out_features) 
	: weight_(new Tensor<float_t>(Shape{1, out_features, in_features}, true)),
	  bias_(new Tensor<float_t>(Shape{1, out_features, 1}, true)) {}

Node<float_t> Linear::forward(const Node<float_t>& input) {
	// (batch, in) <unsqueeze> ==> (batch, in, 1)
	// (1, out, in) <bmm> (batch, in, 1) ==> (batch, out, 1)
	// (batch, out, 1) <+> (1, out, 1) ==> (batch, out, 1)
	Node<float_t> unsqueeze_input(input.get_tensor().unsqueeze_(2));
	auto linear_node = op::bmm(weight_, unsqueeze_input) + bias_;
    Tensor<float_t>* result = new Tensor<float_t>(Shape(linear_node.get_exp()), true);
    *result = linear_node;
    return Node<float_t>(result->squeeze_());
}

NamedParamMap Linear::parameters(const std::string& name) {
    return NamedParamMap{
                {name + "_weight", weight_}, 
                {name + "_bias", bias_}};
}

}  // namespace nn
}  // namespace el