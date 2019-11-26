#include "optimizer.h"

namespace el {
namespace nn {
namespace optim {

SGD::SGD(NamedParamMap params, float_t lr)
	: params_(params.begin(), params.end()),
	  lr_(lr) {}

void SGD::zero_grad(void) {
	for(auto param: params_) {
		auto grad = param.second.get_tensor().grad();
		grad = ConstantExp<float_t>(0, grad.dim());
	}
}

void SGD::step(void) {
	for(auto param: params_) {
		auto tensor = const_cast<Tensor<float_t>&>(param.second.get_tensor());
		tensor += ConstantExp<float_t>(-lr_, tensor.dim()) * tensor.grad();
	}
}


}  // namespace optim
}  // namespace nn
}  // namespace el