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

}  // namespace nn
}  // namespace el

#include "conv.h"
#include "linear.h"
#include "cross_entropy.h"

#endif