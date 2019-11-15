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
	const index_t dsize = 2*3*8*7;
	double data[dsize];
	for(int i = 0; i < dsize; i++)
		data[i] = i * 0.1;
	Tensor<> images(data, {2, 3, 8, 7}, false);

	nn::Conv2d conv(/*in_features=*/3,
					/*out_features=*/3, 
					/*kernel_size=*/{2, 3}, 
					/*stride=*/{3, 2}, 
					/*padding=*/{2, 1});

	cout << "forward" << endl;
	auto result = conv.forward(op::node(images));
	cout << "backward" << endl;
	result.backward();

	auto grad_weight = conv.weight_.get_tensor().grad();
	auto grad_bias = conv.bias_.get_tensor().grad();
	print10(grad_weight);
	print10(grad_bias);
	return 0;
}
