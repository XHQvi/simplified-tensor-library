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
	double data1[16], data2[16];
	for(int i = 0; i < 16; i++) {
		data1[i] = i * 0.1;
		data2[i] = i * 0.2;
	}

	Tensor<3, double> ten1_3d(data1, {2, 2, 4});
	Tensor<3, double> ten2_3d(data2, {2, 2, 4});
	Tensor<3, double> ten3_3d(data1, {2, 1, 4});
	Tensor<3, double> ten_3d({2, 2, 4});
	print10(ten1_3d);print10(ten2_3d);print10(ten3_3d);

	// ten_3d = ten1_3d + ten2_3d; print10(ten_3d);
	// ten_3d = ten1_3d + ten3_3d; print10(ten_3d);
	// ten_3d = ten1_3d + op::sigmoid(ten3_3d); print10(ten_3d);
	ten_3d = op::sigmoid(ten1_3d + ten2_3d); print10(ten_3d);
	ten_3d = op::sigmoid(ten1_3d + ten2_3d) + ten1_3d; print10(ten_3d);
	// ten_3d = ten2_3d - ten1_3d; print10(ten_3d);
	return 0;
}
