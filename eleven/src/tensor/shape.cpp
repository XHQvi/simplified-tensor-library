#include "shape.h"

namespace el {

Shape::Shape(std::initializer_list<index_t> dims)
	: dims_(dims) {}

Shape::Shape(const  Shape& other, index_t skip)
	: dims_(other.dim() - 1) {
	index_t i = 0;
	for(; i < dims_.size() && i != skip; i++)
		dims_[i] = other[i];
	for(; i < dims_.size(); i++)
		dims_[i] = other[i+1];
}

Shape::Shape(index_t value, index_t dim)
	: dims_(dim) {
	for(index_t i = 0; i < dim; i++)
		dims_[i] = value;
}

index_t Shape::dsize() const {
	index_t ds = 1;
	for(index_t i = 0; i < dims_.size(); i++)
		ds *= dims_[i];
	return ds;
}

index_t Shape::subsize(index_t start_dim, index_t end_dim) const {
	index_t ds = 1;
	for(; start_dim < end_dim; start_dim++)
		ds *= dims_[start_dim];
	return ds;
}

index_t Shape::subsize(index_t start_dim) const {
	return subsize(start_dim, dims_.size());
}

std::ostream& operator<<(std::ostream& out, const Shape& s) {
    out << s.dims_;
    return out;
}

}  // namespace el