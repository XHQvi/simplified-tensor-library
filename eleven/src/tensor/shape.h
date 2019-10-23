#ifndef TENSOR_SHAPE_H_
#define TENSOR_SHAPE_H_

#include <iostream>
#include <initializer_list>
#include "../base/type.h"

namespace el{

template<index_t Dim>
struct Shape {
    const static index_t DIM = Dim;
    index_t dims_[Dim];

    // constructor
    Shape(std::initializer_list<index_t> dims);
    Shape(const Shape& other) = default;
    Shape(const Shape<Dim+1>& other, index_t skip);

    // method
    index_t dsize() const;
    index_t subsize(index_t start_dim=0, index_t end_dim=Dim) const;
    index_t dim() const;
    const index_t& operator[](index_t idx) const;
    index_t& operator[](index_t idx);

    //friend
    template<index_t Dim1> friend std::ostream& operator<<(std::ostream& out, const Shape<Dim1>& s);
};


template<index_t Dim>
Shape<Dim>::Shape(std::initializer_list<index_t> dims) {
    index_t i = 0;
    for(auto d: dims) {
        dims_[i++] = d;
        if(i >= DIM) break;
    }
    while(i < DIM)
        dims_[i++] = 1;
}

template<index_t Dim>
Shape<Dim>::Shape(const Shape<Dim+1>& other, int skip) {
    int i = 0;
    for(i = 0; i != skip && i < Dim; i++) 
        dims_[i] = other.dims_[i];
    for(; i < Dim; i++)
        dims_[i] = other.dims_[i+1];
}

template<index_t Dim>
index_t Shape<Dim>::dsize() const {
    int ds = 1;
    for(int i = 0; i < DIM; i++)
        ds *= dims_[i];
    return ds;
}

template<index_t Dim>
index_t Shape<Dim>::subsize(index_t start_dim, index_t end_dim) const {
    index_t ds = 1;
    for(; start_dim < end_dim; start_dim++)
        ds *= dims_[start_dim];
    return ds;
}

template<index_t Dim>
inline index_t Shape<Dim>::dim() const {return DIM;}

template<index_t Dim>
inline const index_t& Shape<Dim>::operator[](index_t idx) const {return dims_[idx];}

template<index_t Dim>
inline index_t& Shape<Dim>::operator[](index_t idx) {return dims_[idx];}

template<index_t Dim>
std::ostream& operator<<(std::ostream& out, const Shape<Dim>& s) {
    out << "Shape(";
    for(int i = 0; i < s.dim(); i++)
        out << s.dims_[i] << ", ";
    out << ')';
}

}  // namespace el
#endif // TENSOR_SHAPE_H_
