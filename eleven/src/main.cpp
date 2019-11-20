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

template<typename Dtype>
void print10(Dtype t) {
	cout << "scalar(" << t << ')' << endl;
}

int main()
{
	// const index_t dsize = 2*3*8*7;
	// double data[dsize];
	// for(int i = 0; i < dsize; i++)
	// 	data[i] = i * 0.1;
	// Tensor<> images(data, {2, 3, 8, 7}, true);
	// print10(images);

	// nn::MaxPool2D pool({2, 2});
	// auto result = pool.forward(op::node(images));

	// print10(result.get_tensor());
	// result.backward();
	// print10(images.grad());

	// auto result_node = op::sum(op::node(images), 2);
	// Tensor<> result(Shape(result_node.get_exp()), true);
	// result = result_node;

	// print10(result);
	// op::node(result).backward();
	// print10(images.grad());

	double data[50];
	for(int i = 0; i < 50; i++)
		data[i] = (double)i*i/50.;
	int ldata[5] = {0, 2, 5, 9, 8};
	Tensor<double> prob(data, {5, 10}, true);
	Tensor<int> labels(ldata, {5}, false);
	print10(prob);
	print10(labels);

	nn::CrossEntropy criterion;
	auto loss = criterion.forward(op::node(prob), op::node(labels));
	loss.backward();
	print10(loss.get_tensor().item());
	print10(prob.grad());

	return 0;
}
