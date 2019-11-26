#ifndef MODELS_LENET_H_
#define MODELS_LENET_H_

#include "../nn/nn.h"

namespace el {
namespace models {

class LeNet {
public:
	nn::Conv2d conv1;
	nn::MaxPool2D pool1;
	nn::Conv2d conv2;
	nn::MaxPool2D pool2;
	nn::Linear fc1;
	nn::Linear fc2;
	nn::Linear fc3;
	nn::ReLU relu;

	LeNet();
	Node<float_t> forward(const Node<float_t>& inputs);
	nn::NamedParamMap parameters(void);
};


class TripleLinear {
public:
	nn::Linear fc1;
	nn::Linear fc2;
	nn::Linear fc3;
	nn::ReLU relu;

	TripleLinear();
	Node<float_t> forward(const Node<float_t>& inputs);
	nn::NamedParamMap parameters(void);
};

}  // namespace models
}  // namespace el

#endif
