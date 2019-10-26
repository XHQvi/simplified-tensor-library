#include <iostream>
#include "tensor/tensor.h"

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
	double data1[15], data2[15];
	for(int i = 0; i < 15; i++) {
		data1[i] = i * 0.1;
		data2[i] = i * 0.2;
	}

	Tensor<2, double> ten1(data1, {3, 5});
	Tensor<2, double> ten2(data2, {5, 3});
	Tensor<2, double> ten3({3, 3});
	print10(ten1);
	print10(ten2);
	ten3 = op::mm(ten1, ten2);
	print10(ten3);

	return 0;
}
