#include <iostream>
#include "tensor/tensor.h"
#include "nn/conv.h"
#include "expression/op.h"

using namespace std;
using namespace el;

template<typename Dtype>
void print10(const Tensor<Dtype>& t) {
	cout << "=====================" << endl;
	cout << "shape:  " << t.size() << " | ";
	cout << "stride: " << t.stride() << endl;
	cout << "offset: " << t.offset() << " | ";
	cout << "is unchanged: " << (t.is_unchanged() ? "True":"False") << endl;
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

	Tensor<double> ten1(data1, {2, 3, 7, 7});
	Tensor<double> ten2(data2, {2, 3, 7, 7});
	print10(ten1);
	print10(ten2);

	auto node_ten1 = op::node(ten1);
	auto node1 = node_ten1 + op::node(ten2);
	auto node2 = op::node(ten1) + node1;
	print10(node_ten1.get());
	return 0;
}
