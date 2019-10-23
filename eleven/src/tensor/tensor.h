#ifndef TENSOR_TENSOR_H_
#define TENSOR_TENSOR_H_

#include <random>
#include <ctime>
#include "tensor_impl.h"

namespace el{
// declaration
template<index_t Dim, typename Dtype=TENSOR_DEFAULT_TYPE> struct Tensor;

Tensor<1, DataType::int_t> arange(index_t start, index_t end, index_t stride=1);
Tensor<1, DataType::int_t> arange(index_t end);

template<index_t Dim> Tensor<Dim, DataType::float_t> rand(const Shape<Dim>& shape);

} // namespace el

namespace el {
// definition
Tensor<1, DataType::int_t> arange(index_t start, index_t end, index_t stride) {
	index_t dsize = (std::abs(end - start)) / std::abs(stride);

	Storage<index_t> storage{dsize};
	for(index_t i = 0; i < dsize; i++)
		storage[i] = i * stride + start;

	Shape<1> shape{dsize};
	return Tensor<1, index_t>(storage, shape);
}

inline Tensor<1, DataType::int_t> arange(index_t end) {return arange(0, end, 1);}

template<index_t Dim>
Tensor<Dim, DataType::float_t> rand(const Shape<Dim>& shape) {
	static std::default_random_engine e(std::time(0));
	static std::uniform_real_distribution<DataType::float_t> u(0, 1);

	index_t dsize = shape.dsize();
	Storage<DataType::float_t> storage{dsize};
	for(index_t i = 0; i < dsize; i++)
		storage[i] = u(e);
	return Tensor<Dim, DataType::float_t>(storage, shape);
}

} // namespace el


#endif