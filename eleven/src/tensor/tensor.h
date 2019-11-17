#ifndef TENSOR_TENSOR_H_
#define TENSOR_TENSOR_H_

#include "tensor_impl.h"

namespace el{
// declaration
template<typename Dtype=TENSOR_DEFAULT_TYPE> struct Tensor;

Tensor<int_t> arange(index_t start, index_t end, index_t stride=1);
Tensor<int_t> arange(index_t end);
Tensor<float_t> rand(const Shape& shape);
Tensor<float_t> ones(const Shape& shape);
Tensor<float_t> zeros(const Shape& shape);

} // namespace el

#endif