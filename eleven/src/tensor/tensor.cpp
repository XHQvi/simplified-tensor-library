#include "tensor.h"
#include <random>
#include <ctime>

namespace el {

Tensor<int_t> arange(index_t start, index_t end, index_t stride) {
	index_t dsize = (std::abs(end - start)) / std::abs(stride);

	Storage<index_t> storage{dsize};
	for(index_t i = 0; i < dsize; i++)
		storage[i] = i * stride + start;

	Shape shape{dsize};
	return Tensor<index_t>(storage, shape);
}

inline Tensor<int_t> arange(index_t end) {return arange(0, end, 1);}

Tensor<float_t> rand(const Shape& shape) {
	static std::default_random_engine e(std::time(0));
	static std::uniform_real_distribution<float_t> u(0, 1);

	index_t dsize = shape.dsize();
	Storage<float_t> storage{dsize};
	for(index_t i = 0; i < dsize; i++)
		storage[i] = u(e);
	return Tensor<float_t>(storage, shape);
}

Tensor<float_t> ones(const Shape& shape) {
	index_t dsize = shape.dsize();
	Storage<float_t> storage{dsize, 1};
	return Tensor<float_t>(storage, shape);
}

Tensor<float_t> zeros(const Shape& shape) {
	index_t dsize = shape.dsize();
	Storage<float_t> storage{dsize, 0};
	return Tensor<float_t>(storage, shape);	
}

} // namespace el
