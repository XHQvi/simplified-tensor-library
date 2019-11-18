#ifndef NN_CROSS_ENTROPY_H_
#define NN_CROSS_ENTROPY_H_

#include "nn.h"

namespace el {
namespace nn {

class CrossEntropy {
public:
	Node<float_t> forward(const Node<float_t>& inputs, const Node<int_t>& labels);
};

}
}

#endif