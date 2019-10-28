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
	const index_t dsize = 2*3*7*7;
	double data1[dsize], data2[dsize];
	for(int i = 0; i < dsize; i++) {
		data1[i] = i * 0.1;
		data2[i] = i * 0.2;
	}

	Tensor<4, double> images(data1, {2, 3, 7, 7});
	print10(images);


	auto exp = op::img2col(images,	{3, 3}, {1, 1}, {1, 1});
	Tensor<2> cols(Shape<2>{exp.size(0), exp.size(1)});
	cols = exp;
	print10(cols);

	return 0;
}
