#include <iostream>
#include "tensor/tensor.h"
#include "nn/conv.h"

using namespace std;
using namespace el;

template<index_t Dim, typename Dtype>
void print10(const Tensor<Dim, Dtype>& t) {
	cout << "=====================" << endl;
	cout << t.shape_ << endl;
	cout << "stride: ";
	for(int i = 0; i < t.dim(); i++)
		cout << t.stride_[i] << " ";
	cout << endl;
	cout << "offset: " << t.storage_.offset() << endl;
	cout << t << endl;
}

void print10(index_t t) {
	cout << "scalar(" << t << ')' << endl;
}

int main()
{
	const index_t dsize = 2*3*7*7;
	double data1[dsize], data2[dsize];
	for(int i = 0; i < dsize; i++) {
		data1[i] = i * 0.1;
		data2[i] = i * 0.2;
	}

	Tensor<4, double> images(data1, Shape<4>{2, 3, 7, 7});
	print10(images);

	nn::Conv2d<double> conv(3, 3, {2, 2}, {3, 2}, {2, 1});
	for(index_t i = 0; i < conv.weight_.shape_.dsize(); i++)
		conv.weight_.storage_[i] = 1.;
	for(index_t i = 0; i < conv.bias_.shape_.dsize(); i++)
		conv.bias_.storage_[i] = 1.;

	index_t loc[3] = {0, 0, 0};
	conv.weight_.eval(loc) = 2.;
	loc[2] = 3;
	conv.weight_.eval(loc) = 3.;
	loc[2] = 1;
	conv.weight_.eval(loc) = 4.;
	loc[2] = 2;
	conv.weight_.eval(loc) = 6;
	loc[2] = 0; loc[1] = 1;
	conv.bias_.eval(loc) = 2.;

	auto result = conv.forward(images);
	print10(result);

	return 0;
}
