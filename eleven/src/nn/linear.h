#ifndef NN_LINEAR_H_
#define NN_LINEAR_H_

#include "nn.h"

namespace el {
namespace nn {

class Linear {
public:
	Node<float_t> weight_;
	Node<float_t> bias_;

	Linear(index_t in_features, index_t out_features);
	Node<float_t> forward(const Node<float_t>& input);
    const NamedParamMap parameters(void);
};

}  // namespace nn
}  // namespace el

#endif
