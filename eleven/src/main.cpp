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
	auto ten1 = rand(Shape<3>{3, 2, 3});
	auto ten2 = arange(16);
	print10(ten1);
	print10(ten2);
	return 0;
}
