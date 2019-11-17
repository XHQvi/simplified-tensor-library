#ifndef TENSOR_SHAPE_H_
#define TENSOR_SHAPE_H_

#include <iostream>
#include <initializer_list>
#include "../utils/base.h"

namespace el{

template<typename Dtype> class Exp;

class Shape {
public:
    // constructor
    Shape(std::initializer_list<index_t> dims);
    Shape(const Shape& other) = default;
    Shape(const Shape& other, index_t skip);
    Shape(index_t* dims, index_t dim);
    template<typename Dtype> Shape(const Exp<Dtype>& exp);

    // method
    index_t dsize() const;
    index_t subsize(index_t start_dim, index_t end_dim) const;
    index_t subsize(index_t start_dim) const;
    index_t dim(void) const {return dims_.size();}
    index_t operator[](index_t idx) const {return dims_[idx];}
    index_t& operator[](index_t idx) {return dims_[idx];}
    //friend
    friend std::ostream& operator<<(std::ostream& out, const Shape& s);
private:
    IndexArray dims_;
};
 
template<typename Dtype>
Shape::Shape(const Exp<Dtype>& exp) 
    : dims_(exp.dim()) {
    for(index_t i = 0; i < dims_.size(); i++)
        dims_[i] = exp.size(i);
}

}  // namespace el
#endif // TENSOR_SHAPE_H_
