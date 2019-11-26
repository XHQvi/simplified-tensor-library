#include <ctime>
#include <random>
#include "data.h"


namespace el {
namespace data {

std::shared_ptr<index_t> shuffle_indice(index_t num) {
	std::default_random_engine e(std::time(0));

	std::shared_ptr<index_t> indice_ptr(new index_t[num], 
										std::default_delete<index_t[]>());
	auto indice = indice_ptr.get();
	for(index_t i = 0; i < num; i++) 
		indice[i] = i;

	index_t temp, fix;
	for(index_t i = num - 1; i > 0; i--) {
		fix = e() % i;
		temp = indice[fix];
		indice[fix] = indice[i];
		indice[i]  = temp;
	}
	return indice_ptr;
}

void dup_images(const el::float_t* base_images, el::float_t* batch_images, 
	 	        const index_t* ids, index_t batch_size, 
	 	        index_t num_pixels) {
	for(index_t i = 0; i < batch_size; i++) {
		index_t sample_idx = ids[i];
		const el::float_t* base_sample = base_images + sample_idx * num_pixels;
		el::float_t* batch_sample = batch_images + i * num_pixels;
		for(index_t j = 0; j < num_pixels; j++)
			batch_sample[j] = base_sample[j];
	}
}

void dup_labels(const el::int_t* base_labels, el::int_t* batch_labels, 
	 	        const index_t* ids, index_t batch_size) {
	for(index_t i = 0; i < batch_size; i++) {
		index_t sample_idx = ids[i];
		batch_labels[i] = base_labels[sample_idx];
	}
}

}  // namespace data
}  // namespace el