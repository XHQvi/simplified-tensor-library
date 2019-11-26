#ifndef NN_OPTIMIZER_H_
#define NN_OPTIMIZER_H_

#include "nn.h"

namespace el {
namespace nn {
namespace optim {

class SGD {
public:
	float_t lr_;
	NamedParamMap params_;

	SGD(NamedParamMap params, float_t lr);
	void zero_grad(void);
	void step(void);

};
 

}  // namespace optim
}  // namespace nn
}  // namespace el



#endif