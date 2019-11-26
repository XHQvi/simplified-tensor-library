#include "models.h"



namespace el {
namespace models {

TripleLinear::TripleLinear()
	: fc1(784, 512),
	  fc2(512, 512),
	  fc3(512, 10),
	  relu() {}

Node<float_t> TripleLinear::forward(const Node<float_t>& inputs) {
	auto fc1_x = fc1.forward(inputs);
	auto relu1_x = relu.forward(fc1_x);
	auto fc2_x = fc2.forward(relu1_x);
	auto relu2_x = relu.forward(fc2_x);
	auto fc3_x = fc3.forward(relu2_x);
	return fc3_x;
}

nn::NamedParamMap TripleLinear::parameters(void) {
	nn::NamedParamMap params;
	auto fc1_params = fc1.parameters("fc1");
	params.insert(fc1_params.begin(), fc1_params.end());
	auto fc2_params = fc2.parameters("fc2");
	params.insert(fc2_params.begin(), fc2_params.end());
	auto fc3_params = fc3.parameters("fc3");
	params.insert(fc3_params.begin(), fc3_params.end());
	return params;
}

}  // namespace models
}  // namespace el