#ifndef TENSOR_TENSOR_H_
#define TENSOR_TENSOR_H_

#include <random>
#include <ctime>
#include <iostream>
#include "tensor_impl.h"


namespace el{
template<typename Dtype=TENSOR_DEFAULT_TYPE> struct Tensor;

template<typename Dtype>
Tensor<Dtype> arange(index_t start, index_t end, index_t stride=1);

template<typename Dtype>
Tensor<Dtype> arange(index_t end);

template<typename Dtype>
Tensor<Dtype> rand(const Shape& shape);

template<typename Dtype>
Tensor<Dtype> ones(const Shape& shape);

template<typename Dtype>
Tensor<Dtype> zeros(const Shape& shape);

template<typename Dtype>
void print10(const Tensor<Dtype>& t);

template<typename Dtype>
void print10(Dtype t);

}  // namespace el



namespace el {

template<typename Dtype>
Tensor<Dtype> arange(index_t start, index_t end, index_t stride) {
	index_t dsize = (std::abs(end - start)) / std::abs(stride);

	Storage<Dtype> storage{dsize};
	for(index_t i = 0; i < dsize; i++)
		storage[i] = i * stride + start;

	Shape shape{dsize};
	return Tensor<Dtype>(storage, shape);
}

template<typename Dtype>
inline Tensor<Dtype> arange(index_t end) {return arange<Dtype>(0, end, 1);}

template<typename Dtype>
Tensor<Dtype> rand(const Shape& shape) {
	std::default_random_engine e(std::time(0));
	std::uniform_real_distribution<Dtype> u(0, 1);

	index_t dsize = shape.dsize();
	Storage<Dtype> storage{dsize};
	for(index_t i = 0; i < dsize; i++)
		storage[i] = u(e);
	return Tensor<Dtype>(storage, shape);
}

template<typename Dtype>
Tensor<Dtype> ones(const Shape& shape) {
	index_t dsize = shape.dsize();
	Storage<Dtype> storage{dsize, 1};
	return Tensor<Dtype>(storage, shape);
}

template<typename Dtype>
Tensor<Dtype> zeros(const Shape& shape) {
	index_t dsize = shape.dsize();
	Storage<Dtype> storage{dsize, 0};
	return Tensor<Dtype>(storage, shape);	
}

template<typename Dtype>
void print10(const Tensor<Dtype>& t) {
	using std::cout;
	using std::endl;
	cout << "=====================" << endl;
	cout << "shape:  " << t.size() << endl;
	cout << "stride: " << t.stride() << endl;
	cout << "offset: " << t.offset() << endl;
	cout << "version: " << t.version() << endl;
	cout << t << endl;
}

template<typename Dtype>
void print10(Dtype t) {
	using std::cout;
	using std::endl;
	cout << "scalar(" << t << ')' << endl;
}

} // namespace el

#endif