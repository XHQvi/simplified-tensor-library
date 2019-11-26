#ifndef NN_NN_H_
#define NN_NN_H_

#include <map>
#include <string>
#include "../tensor/tensor.h"
#include "../expression/op.h"

namespace el {
namespace nn {

using NamedParamMap = std::map<std::string, Node<float_t>&>;

class Conv2d;
class Linear;
class CrossEntrpy;
class ReLU;
class MaxPool2D;

}  // namespace nn
}  // namespace el
#include "init.h"
#include "optimizer.h"

#include "conv.h"
#include "cross_entropy.h"
#include "linear.h"
#include "max_pool.h"
#include "relu.h"

#endif