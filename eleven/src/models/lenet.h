#ifndef MODELS_LENET_H_
#define MODELS_LENET_H_

#include "../nn/nn.h"

namespace el {
namespace models {

class LeNet {
public:
	nn::Conv2d conv1;
	nn::Conv2d conv2;
	nn::Conv2d conv3;
	nn::Conv2d conv4;
	nn::Conv2d conv5;

	Node<float_t> forward(const Node<float_t>& inputs);
	const nn::NamedParamMap parameters(void);
}

}  // namespace models
}  // namespace el

#endif
