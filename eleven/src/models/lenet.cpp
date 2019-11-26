#include "models.h"

namespace el {
namespace models{

LeNet::LeNet()
	: conv1(3, 6, 5, 1, 0),
	  pool1(2),
	  conv2(6, 16, 5, 1, 0),
	  pool2(2),
	  fc1(400, 120),
	  fc2(120, 84),
	  fc3(84, 10),
	  relu() {}

Node<float_t> LeNet::forward(const Node<float_t>& inputs) {
	index_t batch_size = inputs.size(0);
	auto conv1_x = conv1.forward(inputs);  // b, 6, 28, 28
	auto relu1_x = relu.forward(conv1_x);
	auto pool1_x = pool1.forward(relu1_x);  // b, 6, 14, 14

	auto conv2_x = conv2.forward(pool1_x);  // b, 16, 10, 10
	auto relu2_x = relu.forward(conv2_x);
	auto pool2_x = pool2.forward(relu2_x);  // b, 16, 5, 5

	auto flatten = pool2_x.get_tensor().view_({batch_size, 400});
	auto fc1_x = fc1.forward(op::node(flatten));  // b, 120
	auto relu_fc1_x = relu.forward(fc1_x);

	auto fc2_x = fc2.forward(relu_fc1_x);  // b, 84
	auto relu_fc2_x = relu.forward(fc2_x);

	auto fc3_x = fc3.forward(relu_fc2_x);  // b, 10
	return fc3_x;
}

nn::NamedParamMap LeNet::parameters(void) {
	nn::NamedParamMap params;
	auto conv1_params = conv1.parameters("conv1");
	params.insert(conv1_params.begin(), conv1_params.end());
	auto conv2_params = conv2.parameters("conv2");
	params.insert(conv2_params.begin(), conv2_params.end());
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
