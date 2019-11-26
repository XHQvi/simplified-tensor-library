#ifndef DATA_DATA_H_
#define DATA_DATA_H_

#include <iostream>
#include <string>
#include <fstream>
#include <memory>

#include "../utils/base.h"


namespace el {
namespace data {

std::shared_ptr<int_t> read_mnist_labels(std::string filename);
std::shared_ptr<float_t> read_mnist_images(std::string filename);

std::shared_ptr<index_t> shuffle_indice(index_t num);

void dup_images(const el::float_t* base_data, el::float_t* batch_data, 
	 	        const index_t* ids, index_t batch_size, 
	 	        index_t num_pixels);

void dup_labels(const el::int_t* base_data, el::int_t* batch_data, 
	 	        const index_t* ids, index_t batch_size);

}  // namespace data
}  // namespada el


#endif