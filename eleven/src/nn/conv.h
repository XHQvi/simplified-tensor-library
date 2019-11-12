#ifndef NN_CONV_H_
#define NN_CONV_H_

#include "../tensor/tensor.h"
#include "../expression/op.h"

namespace el {
namespace nn {

class Conv2d {
public:
    // TODO:
    // weight should be maintained by nodes, and the tensor contained by it should require grad.
    Tensor<float_t> weight_;
    Tensor<float_t> bias_;

    Conv2d(index_t in_features, index_t out_features, const std::pair<index_t, index_t>& kernel_size,
           const std::pair<index_t, index_t>& stride, const std::pair<index_t, index_t>& padding)
        : in_features_(in_features), out_features_(out_features), kernel_size_(kernel_size),
          stride_(stride), padding_(padding), 
          weight_({1, out_features, in_features*kernel_size.first*kernel_size.second}),
          bias_({1, out_features, 1}) {}
    Conv2d(index_t in_features, index_t out_features, index_t kernel_size,
           index_t stride, index_t padding)
        : Conv2d(in_features, out_features, {kernel_size, kernel_size}, {stride, stride}, {padding, padding}) {}
    Tensor<float_t> forward(const Tensor<float_t>& imgs) {
        auto col_exp = op::img2col(imgs, kernel_size_, stride_, padding_);
        auto exp = op::bmm(weight_, col_exp) + bias_;
        Tensor<float_t> result({exp.size(0), exp.size(1), exp.size(2)});
        result = exp;
        return result.view({imgs.size(0), out_features_, col_exp.out_size(0), col_exp.out_size(1)});
    }
    Node<float_t> forward(const Node<float_t>& imgs) {
        auto col_node = op::img2col(imgs, kernel_size_, stride_, padding_);
        auto col_exp = col_node.get<op::Img2ColExp>();
        auto conv_node = op::bmm(op::node(weight_), col_node) + op::node(bias_);
        auto conv_exp = conv_node.get<op::AddExp>();
        Tensor<float_t>* result = new Tensor<float_t>({conv_exp.size(0), conv_exp.size(1), conv_exp.size(2)}, true);
        *result = conv_node;
        // TODO: There are unnecessary construction of tensors.
        return Node<float_t>(new Tensor<float_t>(
            result->view({imgs.size(0), out_features_, col_exp.out_size(0), col_exp.out_size(1)})));
    }
private:
    index_t in_features_, out_features_;
    std::pair<index_t, index_t> kernel_size_;
    std::pair<index_t, index_t> stride_;
    std::pair<index_t, index_t> padding_;
};

}  // namespace nn
}  // namespace el
#endif