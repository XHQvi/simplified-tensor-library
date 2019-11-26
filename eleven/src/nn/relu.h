#ifndef NN_RELU_H_
#define NN_RELU_H_

#include "nn.h"

namespace el {
namespace nn{

class ReLU {
public:
	Node<float_t> forward(const Node<float_t>& inputs);
};

}  // namespace nn
}  // namespace el

#endif