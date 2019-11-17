#include <iostream>
#include "tensor/tensor.h"
#include "expression/op.h"
#include "nn/conv.h"
#include "nn/init.h"
#include "nn/linear.h"

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
	nn::Linear linear(/*in_features=*/48,
					  /*out_features=*/10);
	nn::init::constant_init(conv.parameters(), 1);
	nn::init::constant_init(linear.parameters(), 1);

	cout << "forward" << endl;
	auto conv_ret = conv.forward(op::node(images));
	auto view_ret = op::node(conv_ret.get_tensor().view_({2, 48}));
	auto linear_ret = linear.forward(view_ret);

/* ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */
	nn::Conv2d conv1(/*in_features=*/3,
					/*out_features=*/3, 
					/*kernel_size=*/{2, 3}, 
					/*stride=*/{3, 2}, 
					/*padding=*/{2, 1});
	nn::Linear linear1(/*in_features=*/48,
					  /*out_features=*/10);
	nn::init::constant_init(conv1.parameters(), 1);
	nn::init::constant_init(linear1.parameters(), 1);

	cout << "forward" << endl;
	auto conv_ret1 = conv1.forward(op::node(images));
	auto view_ret1 = op::node(conv_ret1.get_tensor().view_({2, 48}));
	auto linear_ret1 = linear1.forward(view_ret1);
/* ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */

	auto add_ret = (linear_ret1 + linear_ret) * linear_ret;
	Tensor<>* result = new Tensor<>(Shape(add_ret.get_exp()), true);
	*result = add_ret;
	cout << "backward" << endl;
	op::node(result).backward();

	auto grad_weight = conv.weight_.get_tensor().grad();
	auto grad_bias = conv.bias_.get_tensor().grad();
	print10(grad_weight);
	print10(grad_bias);
	auto linear_grad_weight = linear.weight_.get_tensor().grad();
	auto linear_grad_bias = linear.bias_.get_tensor().grad();
	print10(linear_grad_weight);
	print10(linear_grad_bias);

	return 0;
}
