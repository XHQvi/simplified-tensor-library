#ifndef NN_MAX_POOLING_H_
#define NN_MAX_POOLING_H_

#include "nn.h"

namespace el {
namespace nn {

class MaxPool2D {
public:
	MaxPool2D(const std::pair<index_t, index_t>& kernel_size);
	Node<float_t> forward(const Node<float_t>& inputs);
private:
	std::pair<index_t, index_t> kernel_size_;
};

}
}

#endif