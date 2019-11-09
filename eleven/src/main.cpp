#include <iostream>
#include "tensor/tensor.h"
#include "nn/conv.h"
#include "expression/op.h"

using namespace std;
using namespace el;

template<typename Dtype>
void print10(const Tensor<Dtype>& t) {
	cout << "=====================" << endl;
	cout << "shape:  " << t.size() << endl;
	cout << "stride: " << t.stride() << endl;
	cout << "offset: " << t.offset() << endl;
	cout << "version: " << t.version() << endl;
	cout << t << endl;
}

void print10(index_t t) {
	cout << "scalar(" << t << ')' << endl;
}

int main()
{
{
	const index_t dsize = 2*3*8*7;
	double data[dsize];
	for(int i = 0; i < dsize; i++)
		data[i] = i * 0.1;
	Tensor<> images(data, {2, 3, 8, 7}, true);

	nn::Conv2d conv(/*in_features=*/3,
					/*out_features=*/3, 
					/*kernel_size=*/{2, 3}, 
					/*stride=*/{3, 2}, 
					/*padding=*/{2, 1});
	cout << conv.weight_.size() << endl;
	cout << conv.bias_.size() << endl;

	double weight_data[54], bias_data[3];
	for(int i = 0; i < 54; i++)
		weight_data[i] = i * 0.01;
	for(int i = 0; i < 3; i++)
		bias_data[i] = i * 0.05;
	conv.weight_ = Tensor<>(weight_data, {1, 3, 18});
	conv.bias_ = Tensor<>(bias_data, {1, 3, 1});

	auto result = conv.forward(op::node(images));
	// print10(result.get_tensor());
	result.backward();
}

	return 0;
}
