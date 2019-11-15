#ifndef NN_CONV_H_
#define NN_CONV_H_

#include "../tensor/tensor.h"
#include "../expression/op.h"

namespace el {
namespace nn {

class Conv2d {
public:
    Node<float_t> weight_;
    Node<float_t> bias_;

    Conv2d(index_t in_features, 
           index_t out_features, 
           const std::pair<index_t, index_t>& kernel_size,
           const std::pair<index_t, index_t>& stride, 
           const std::pair<index_t, index_t>& padding);

    Conv2d(index_t in_features, 
           index_t out_features, 
           index_t kernel_size,
           index_t stride, 
           index_t padding);

    Node<float_t> forward(const Node<float_t>& imgs);
private:
    index_t in_features_, out_features_;
    std::pair<index_t, index_t> kernel_size_;
    std::pair<index_t, index_t> stride_;
    std::pair<index_t, index_t> padding_;
};

}  // namespace nn
}  // namespace el
#endif