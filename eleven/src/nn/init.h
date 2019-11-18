#ifndef NN_INIT_H_
#define NN_INIT_H_

#include "nn.h"


namespace el {
namespace nn {
namespace init {


void uniform_init(const NamedParamMap& params);
void uniform_init(Node<float_t>& param);

void constant_init(const NamedParamMap& params, float_t value);
void constant_init(Node<float_t>& param, float_t value);

}  // namespace init
}  // namespace nn
}  // namespace el
#endif