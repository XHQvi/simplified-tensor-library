#ifndef NN_CONV_H_
#define NN_CONV_H_

#include "../base/type.h"
#include "../tensor/tensor.h"
#include "../expression/expression.h"

namespace el {
namespace nn {

template<typename Dtype>
class Conv2d {
public:
    index_t in_features_, out_features_;
    std::pair<index_t, index_t> kernel_size_;
    std::pair<index_t, index_t> stride_;
    std::pair<index_t, index_t> padding_;
    Tensor<3, Dtype> weight_;
    Tensor<3, Dtype> bias_;

// public:
    Conv2d(index_t in_features, index_t out_features, const std::pair<index_t, index_t>& kernel_size,
           const std::pair<index_t, index_t>& stride, const std::pair<index_t, index_t>& padding)
        : in_features_(in_features), out_features_(out_features), kernel_size_(kernel_size),
          stride_(stride), padding_(padding), 
          weight_({1, out_features, in_features*kernel_size.first*kernel_size.second}),
          bias_({1, out_features, 1}) {}
    Conv2d(index_t in_features, index_t out_features, index_t kernel_size,
           index_t stride, index_t padding)
        : in_features_(in_features), out_features_(out_features), kernel_size_({kernel_size, kernel_size}),
          stride_({stride, stride}), padding_({padding, padding}),
          weight_({1, out_features, in_features*kernel_size*kernel_size}),
          bias_({1, out_features, 1}) {}
    Tensor<4, Dtype> forward(const Tensor<4, Dtype>& imgs) {
        auto col_exp = op::img2col(imgs, kernel_size_, stride_, padding_);
        auto exp = op::bmm(weight_, col_exp) + bias_;
        Tensor<3, Dtype> result({exp.size(0), exp.size(1), exp.size(2)});
        result = exp;
        return result.view(Shape<4>{imgs.size(0), out_features_, col_exp.out_size_.first, col_exp.out_size_.second});
    }
};

}  // namespace nn
}  // namespace el
#endif