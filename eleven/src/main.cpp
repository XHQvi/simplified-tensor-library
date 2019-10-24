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

template<typename Dtype>
void print10(const Tensor<1, Dtype>& t) {
	cout << "=====================" << endl;
	cout << "Shape(" << t.shape_ << ", )" << endl;
	cout << "stride: " << t.stride_ << endl;
	cout << endl;
	cout << "offset: " << t.storage_.offset() << endl;
	cout << t << endl;
}

void print10(index_t t) {
	cout << "scalar(" << t << ')' << endl;
}

int main()
{
	double data1[8], data2[8];
	for(int i = 0; i < 8; i++) {
		data1[i] = i * 0.5;
		data2[i] = i * 0.7;
	}

	Tensor<1, double> ten1(data1, {8});
	Tensor<1, double> ten2(data2, {8});
	Tensor<1, double> ten3({8});

	print10(ten1);
	print10(ten2);
	ten3 = ten1 + ten2;
	print10(ten3);
	ten3 = op::sigmoid(ten1);
	print10(ten3);
	ten3 = ten1 + op::sigmoid(ten1);
	print10(ten3);

	return 0;
}
