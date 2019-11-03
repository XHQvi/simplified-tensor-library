#ifndef TENSOR_SHAPE_H_
#define TENSOR_SHAPE_H_

#include <iostream>
#include <initializer_list>
#include "../utils/base.h"

namespace el{

class Shape {
public:
    // constructor
    Shape(std::initializer_list<index_t> dims);
    Shape(const Shape& other) = default;
    Shape(const Shape& other, index_t skip);
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
 
}  // namespace el
#endif // TENSOR_SHAPE_H_
