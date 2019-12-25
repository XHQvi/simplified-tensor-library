#include "init.h"
#include <random>
#include <ctime>


namespace el {
namespace nn {
namespace init {

void uniform_init(Node<float_t>& param, float_t a, float_t b) {
	static std::default_random_engine e(std::time(0));
	std::uniform_real_distribution<float_t> u(a, b);

	Tensor<float_t>& tensor = const_cast<Tensor<float_t>&>(param.get_tensor());
	index_t dsize = tensor.size().dsize();
	for(index_t i = 0; i < dsize; i++)
		tensor.eval(i) = u(e);
}

void uniform_init(const NamedParamMap& params, float_t a, float_t b) {
	for(auto param: params)
		uniform_init(param.second, a, b);
}

void constant_init(Node<float_t>& param, float_t value) {
	Tensor<float_t>& tensor = const_cast<Tensor<float_t>&>(param.get_tensor());
	index_t dsize = tensor.size().dsize();
	for(index_t i = 0; i < dsize; i++)
		tensor.eval(i) = value;
}

void constant_init(const NamedParamMap& params, float_t value) {
	for(auto param: params)
		constant_init(param.second, value);
}

}  // namespace init
}  // namespace nn
}  // namespace el