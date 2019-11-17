#include "conv.h"

namespace el {
namespace nn {

Conv2d::Conv2d(index_t in_features, 
			   index_t out_features, 
			   const std::pair<index_t, index_t>& kernel_size,
       	 const std::pair<index_t, index_t>& stride, 
       	 const std::pair<index_t, index_t>& padding)
    : in_features_(in_features), 
      out_features_(out_features), 
      kernel_size_(kernel_size),
      stride_(stride), 
      padding_(padding), 
      weight_(new Tensor<float_t>(Shape{1, out_features, in_features*kernel_size.first*kernel_size.second}, true)),
      bias_(new Tensor<float_t>(Shape{1, out_features, 1}, true)) {}

Conv2d::Conv2d(index_t in_features, 
	           index_t out_features, 
	           index_t kernel_size,
               index_t stride, 
               index_t padding)
    : Conv2d(in_features, 
    		 out_features, 
    		 {kernel_size, kernel_size}, 
    		 {stride, stride}, 
    		 {padding, padding}) {}

Node<float_t> Conv2d::forward(const Node<float_t>& imgs) {
    auto col_node = op::img2col(imgs, kernel_size_, stride_, padding_);
    auto col_exp = col_node.get<op::Img2ColExp>();
    auto conv_node = op::bmm(weight_, col_node) + bias_;
    Tensor<float_t>* result = new Tensor<float_t>(Shape(conv_node.get_exp()), true);
    *result = conv_node;
    // The result tensor would be maintained by another tensor's next_exp_ which is ConstExptr.
    // So don't worry. It'll be deconstructed at a proper time.
    return Node<float_t>(
        result->view_({imgs.size(0), out_features_, col_exp.out_size(0), col_exp.out_size(1)}));
}

const NamedParamMap Conv2d::parameters(void) {
    return std::map<std::string, Node<float_t>&>{
                {"weight", weight_}, 
                {"bias", bias_}};
}

} // namespace nn
} // namespace el