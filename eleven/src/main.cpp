#include <iostream>
#include "tensor/tensor.h"
#include "expression/op.h"
#include "nn/conv.h"
#include "nn/init.h"
#include "nn/linear.h"

using std::cout;
using std::endl;
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
	// const index_t dsize = 2*3*8*7;
	// double data[dsize];
	// for(int i = 0; i < dsize; i++)
	// 	data[i] = i * 0.1;
	// Tensor<> images(data, {2, 3, 8, 7}, false);

	// nn::Conv2d conv(/*in_features=*/3,
	// 				/*out_features=*/3, 
	// 				/*kernel_size=*/{2, 3}, 
	// 				/*stride=*/{3, 2}, 
	// 				/*padding=*/{2, 1});
	// nn::Linear linear(/*in_features=*/48,
	// 				  /*out_features=*/10);
	// nn::init::constant_init(conv.parameters(), 1);
	// nn::init::constant_init(linear.parameters(), 1);

	double data[5] = {0, 1, 2, 3, 4};
	Tensor<double> origin(data, {5}, true);
	auto log_softmax_ret = op::log_softmax(op::node(origin));

	Tensor<double> result(Shape(log_softmax_ret.get_exp()), true);
	result = log_softmax_ret;
	print10(result);

	op::node(result).backward();
	print10(origin.grad());
	return 0;
}
