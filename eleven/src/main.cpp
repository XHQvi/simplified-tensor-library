#include <iostream>
#include "tensor/tensor.h"
#include "expression/op.h"
#include "nn/nn.h"

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

	double data[30];
	for(index_t i = 0; i < 30; i ++) 
		data[i] = (double)i*i/50.;
	int ldata[3] = {4, 9, 0};
	Tensor<double> prob(data, {3, 10}, true);
	Tensor<int> labels(ldata, {3}, false);
	print10(prob);

	nn::CrossEntropy criterion;
	auto loss = criterion.forward(op::node(prob), op::node(labels));

	print10(loss.get_tensor());
	loss.backward();
	print10(prob.grad());

	return 0;
}
