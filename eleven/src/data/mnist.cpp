#include "data.h"

namespace el {
namespace data {

int ReverseInt(int i) {
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

std::shared_ptr<int_t> read_mnist_labels(std::string filename) {
	using namespace std;

	ifstream file(filename, ios::binary);
	if(!file.is_open())
		THROW_ERROR(NotImplementError, "MNIST labels file can't open.");
	int magic_number, num_images;
	file.read((char*)&magic_number, sizeof(magic_number));
	file.read((char*)&num_images, sizeof(num_images));
	magic_number = ReverseInt(magic_number);
	num_images = ReverseInt(num_images);

	shared_ptr<index_t> labels_ptr(new int_t[num_images], 
								   std::default_delete<int_t[]>());
	auto labels = labels_ptr.get();
	for(index_t i = 0; i < num_images; i++) {
		unsigned char buffer;
		file.read((char*)&buffer, sizeof(buffer));
		labels[i] = buffer;
	}
	cout << "magic_number = " << magic_number << endl;
	cout << "number of images = " << num_images << endl;
	return labels_ptr;
}

std::shared_ptr<float_t> read_mnist_images(std::string filename) {
	using namespace std;

	ifstream file(filename, ios::binary);
	if(!file.is_open())
		THROW_ERROR(NotImplementError, "MNIST images file can't open.");
	int magic_number, num_images, num_rows, num_cols;
	file.read((char*)&magic_number, sizeof(magic_number));
	file.read((char*)&num_images, sizeof(num_images));
	file.read((char*)&num_rows, sizeof(num_rows));
	file.read((char*)&num_cols, sizeof(num_cols));
	magic_number = ReverseInt(magic_number);
	num_images = ReverseInt(num_images);
	num_rows = ReverseInt(num_rows);
	num_cols = ReverseInt(num_cols);

	index_t num_pixels = num_images * num_rows * num_cols;
	shared_ptr<float_t> images_ptr(new float_t[num_pixels], 
								   std::default_delete<float_t[]>());
	auto images = images_ptr.get();
	for(index_t i = 0; i < num_pixels; i++) {
		unsigned char buffer;
		file.read((char*)&buffer, sizeof(buffer));
		images[i] = (float_t)buffer / 255.;
	}
	cout << "magic_number = " << magic_number << endl;
	cout << "number of images = " << num_images << endl;
	cout << "size of images = (" << num_rows << ", "; 
	cout << num_cols << ')' << endl;
	return images_ptr;	
}


}  // namespace data
}  // namespace el 