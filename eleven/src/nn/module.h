#ifndef NN_MODULE_H_
#define NN_MODULE_H_

#include <map>
#include <string>
#include "../tensor/tensor.h"
#include "../expression/op.h"


namespace el {
namespace nn {

using NamedParamMap = std::map<std::string, Node<float_t>&>;

class Module {
public:
    virtual Node<float_t> forward(const Node<float_t>& input) = 0;
    virtual const NamedParamMap parameters(void) = 0;
};

}  // namespace nn
}  // namespace el
#endif
