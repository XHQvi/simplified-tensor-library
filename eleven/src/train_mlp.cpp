#include <iostream>

#include "tensor/tensor.h"
#include "expression/op.h"
#include "nn/nn.h"
#include "models/models.h"
#include "data/data.h"

using std::cout;
using std::endl;
using std::string;
using std::shared_ptr;
using std::make_shared;

using namespace el;

string train_images_path = "D:\\somethingElse\\other_projects\\datasets\\MNIST\\raw\\train-images-idx3-ubyte";
string train_labels_path = "D:\\somethingElse\\other_projects\\datasets\\MNIST\\raw\\train-labels-idx1-ubyte";
string test_images_path = "D:\\somethingElse\\other_projects\\datasets\\MNIST\\raw\\t10k-images-idx3-ubyte";
string test_labels_path = "D:\\somethingElse\\other_projects\\datasets\\MNIST\\raw\\t10k-labels-idx1-ubyte";

index_t train_one_epoch(models::TripleLinear& net,
						nn::CrossEntropy& criterion,
						nn::optim::SGD& optimizer,
						const shared_ptr<el::float_t>& images_ptr,
						const shared_ptr<el::int_t>& labels_ptr,
						index_t num_images,
						index_t batch_size,
						index_t num_pixels) {
	auto data_indice_ptr = data::shuffle_indice(num_images);

	index_t batch_pixel_size = batch_size * num_pixels;
	shared_ptr<el::float_t> batch_images(new el::float_t[batch_pixel_size](),
	    								 std::default_delete<el::float_t[]>());
	shared_ptr<el::int_t> batch_labels(new el::int_t[batch_size](),
									   std::default_delete<el::int_t[]>());

	index_t iter = 0;
	index_t num_iters = (num_images + batch_size - 1) / batch_size;
	for(; iter < num_iters; iter++) {
		index_t this_batch_size = batch_size;
		if(iter == num_iters - 1)
			this_batch_size = num_images - batch_size*iter;

		data::dup_images(images_ptr.get(), batch_images.get(),
					     data_indice_ptr.get() + iter*batch_size, 
					     this_batch_size, num_pixels);
		data::dup_labels(labels_ptr.get(), batch_labels.get(), 
					     data_indice_ptr.get() + iter*batch_size, 
					     this_batch_size);

		Tensor<el::float_t> batch_images_tensor(batch_images.get(), 
			                                    {this_batch_size, num_pixels});
		Tensor<el::int_t> batch_labels_tensor(batch_labels.get(), 
			                                  {this_batch_size});

		auto output = net.forward(op::node(batch_images_tensor));
		auto loss = criterion.forward(output, op::node(batch_labels_tensor));
		
		optimizer.zero_grad();
		loss.backward();
		optimizer.step();

		if(iter % 10 == 0) {
			cout << "iter: " << iter << "/" << num_iters;
			cout << " | loss: " << loss.get_tensor().item() << endl;
		}
	}
	return iter;

}

index_t validate(models::TripleLinear& net,
				 const shared_ptr<el::float_t>& images_ptr,
				 const shared_ptr<el::int_t>& labels_ptr,
				 index_t num_images,
				 index_t batch_size,
				 index_t num_pixels) {
	auto data_indice_ptr = data::shuffle_indice(num_images);

	index_t batch_pixel_size = batch_size * num_pixels;
	shared_ptr<el::float_t> batch_images(new el::float_t[batch_pixel_size](),
	    								 std::default_delete<el::float_t[]>());
	shared_ptr<el::int_t> batch_labels(new el::int_t[batch_size](),
									   std::default_delete<el::int_t[]>());

	index_t iter = 0;
	index_t num_iters = (num_images + batch_size - 1) / batch_size;
	index_t acc = 0;
	for(; iter < num_iters; iter++) {
		index_t this_batch_size = batch_size;
		if(iter == num_iters - 1)
			this_batch_size = num_images - batch_size*iter;

		data::dup_images(images_ptr.get(), batch_images.get(),
					     data_indice_ptr.get() + iter*batch_size, 
					     this_batch_size, num_pixels);
		data::dup_labels(labels_ptr.get(), batch_labels.get(), 
					     data_indice_ptr.get() + iter*batch_size, 
					     this_batch_size);

		Tensor<el::float_t> batch_images_tensor(batch_images.get(), 
			                                    {this_batch_size, num_pixels});
		auto output = net.forward(op::node(batch_images_tensor));
		
		Tensor<double> predict(Shape{output.size(0)});
		predict = op::argmax(output, 1);
		for(index_t i = 0; i < this_batch_size; i++)
			if(predict[{i}] == batch_labels.get()[i])
				acc++;
	}
	cout << "acc of test images: " << acc << " / " << num_images;
	cout << " = " << (double)acc / (double)num_images << endl;
	return iter;

}

int main() {
	cout << "read train data ..." << endl;
	auto train_images_ptr = data::read_mnist_images(train_images_path);
	auto train_labels_ptr = data::read_mnist_labels(train_labels_path);
	cout << "read test data ..." << endl;
	auto test_images_ptr = data::read_mnist_images(test_images_path);
	auto test_labels_ptr = data::read_mnist_labels(test_labels_path);
	index_t num_train_images = 60000, num_test_images = 10000;
	index_t batch_size = 64;
	index_t num_pixels = 28 * 28;

	models::TripleLinear net;
	nn::CrossEntropy criterion;
	nn::optim::SGD optimizer(net.parameters(), 0.05);

	for(index_t epoch = 0; epoch < 1; epoch ++) {
		cout << "***** epoch " << epoch << " train *****" << endl;
		train_one_epoch(net, criterion, optimizer,
						train_images_ptr, train_labels_ptr,
						num_train_images, batch_size, 
						num_pixels);
		validate(net, test_images_ptr, test_labels_ptr,
				 num_test_images, batch_size, num_pixels);
		optimizer.lr_ *= 0.1;
	}
	return 0;
}
