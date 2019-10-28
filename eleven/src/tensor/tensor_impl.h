#ifndef TENSOR_TENSOR_IMPL_H_
#define TENSOR_TENSOR_IMPL_H_

#include <iostream>
#include <initializer_list>
#include "../base/type.h"
#include "../base/exception.h"
#include "storage.h"
#include "shape.h"
#include "../expression/expression.h"

namespace el {
// general declaration
template<index_t Dim, typename Dtype>
struct Tensor: public Exp<Dtype> {
    Storage<Dtype> storage_;
    Shape<Dim> shape_;
    index_t stride_[Dim];

    // constructor
    Tensor(const Storage<Dtype>& storage, const Shape<Dim>& shape, const int* stride)
        : storage_(storage), shape_(shape) {
        for(index_t i = 0; i < shape_.dim(); i++)
            stride_[i] = stride[i];
    }
    Tensor(const Storage<Dtype>& storage, const Shape<Dim>& shape): storage_(storage), shape_(shape) {
        for(index_t i = 0; i < shape_.dim(); i++) {
            if(shape_[i] == 1) stride_[i] = 0; // for broadcasting
            else stride_[i] = shape_.subsize(i + 1);
        }
    }
    Tensor(const Dtype* data, const Shape<Dim>& shape): Tensor(Storage<Dtype>(data, shape.dsize()), shape){}
    explicit Tensor(const Shape<Dim>& shape): Tensor(Storage<Dtype>(shape.dsize()), shape) {}
    template<index_t Dim1> Tensor(const Tensor<Dim1, Dtype>& other, const Shape<Dim>& shape): Tensor(other.storage_, shape) {}
    Tensor(const Tensor& other) = default;
    Tensor(Tensor&& other) = default;

    // method
    index_t dim(void) const;
    index_t size(index_t idx) const;
    Dtype& operator[](std::initializer_list<index_t> ids);
    const Dtype& operator[](std::initializer_list<index_t> ids) const;
    Tensor<Dim-1, Dtype> slice(index_t idx, index_t dim=0) const;
    Tensor<Dim, Dtype> slice(index_t start_idx, index_t end_idx, index_t dim) const;
    Tensor<Dim, Dtype> transpose(index_t dim1, index_t dim2) const;
    bool is_contiguous(void) const;
    template<index_t Dim1> Tensor<Dim1, Dtype> view(const Shape<Dim1>& shape) const;
    // method for implementing template expression
    Dtype eval(index_t* ids) const;
    Dtype& eval(index_t* ids);
    Tensor<Dim, Dtype>& operator=(const Exp<Dtype>& src);
    // friend
    template<index_t Dim1, typename Dtype1> friend std::ostream& operator<<(std::ostream& out, const Tensor<Dim1, Dtype1>& t);
};
} // namespace el (general declaration)

namespace el {
// general definition
template<index_t Dim, typename Dtype>
inline index_t Tensor<Dim, Dtype>::dim(void) const {return shape_.dim();}

template<index_t Dim, typename Dtype>
inline index_t Tensor<Dim, Dtype>::size(index_t idx) const {CHECK_DIM_EXIST(idx, 0, shape_.dim()); return shape_[idx];}

template<index_t Dim, typename Dtype>
inline Tensor<Dim-1, Dtype> Tensor<Dim, Dtype>::slice(index_t idx, index_t dim) const {
    CHECK_INDEX_IN_RANGE(idx, 0, size(dim), dim);

    Storage<Dtype> storage(storage_, stride_[dim] * idx);
    Shape<Dim-1> shape(shape_, dim);
    int i, stride[Dim-1];
    for(i = 0; i != dim && i < Dim-1; i++)
        stride[i] = stride_[i];
    for(;i < Dim-1; i++)
        stride[i] = stride_[i+1];
    return Tensor<Dim-1, Dtype>(storage, shape, stride);
}

template<index_t Dim, typename Dtype>
inline Tensor<Dim, Dtype> Tensor<Dim, Dtype>::slice(index_t start_idx, index_t end_idx, index_t dim) const {
    CHECK_INDEX_IN_RANGE(start_idx, 0, size(dim), dim);
    CHECK_INDEX_IN_RANGE(end_idx, start_idx, size(dim), dim);

    Storage<Dtype> storage(storage_, stride_[dim] * start_idx);
    Shape<Dim> shape(shape_);
    shape.dims_[dim] = end_idx - start_idx;
    return Tensor<Dim, Dtype>(storage, shape, stride_);
}

template<index_t Dim, typename Dtype>
inline Dtype& Tensor<Dim, Dtype>::operator[](std::initializer_list<index_t> ids) {
    CHECK_DIM_MATCH(ids.size(), dim());

    index_t offset = 0, i = 0;
    for(auto idx: ids) {
        offset += idx * stride_[i];
        i++;
    }
    return storage_[offset];
}

template<index_t Dim, typename Dtype>
inline const Dtype& Tensor<Dim, Dtype>::operator[](std::initializer_list<index_t> ids) const {
    CHECK_DIM_MATCH(ids.size(), dim());

    index_t offset = 0, i = 0;
    for(auto idx: ids) {
        offset += idx * stride_[i];
        i++;
    }
    return storage_[offset];
}

template<index_t Dim, typename Dtype>
inline Tensor<Dim, Dtype> Tensor<Dim, Dtype>::transpose(index_t dim1, index_t dim2) const {
    CHECK_DIM_EXIST(dim1, 0, dim());
    CHECK_DIM_EXIST(dim2, 0, dim());

    Tensor<Dim, Dtype> other(*this);
    index_t temp = other.stride_[dim1];
    other.stride_[dim1] = other.stride_[dim2];
    other.stride_[dim2] = temp;

    temp = other.shape_[dim1];
    other.shape_[dim1] = other.shape_[dim2];
    other.shape_[dim2] = temp;
    return other;
}

template<index_t Dim, typename Dtype>
inline bool Tensor<Dim, Dtype>::is_contiguous(void) const {
    for(index_t i = 1; i <= shape_.dim(); i++)
        if(stride_[i - 1] != shape_.subsize(i))
            return false;
    return true;
}

template<index_t Dim, typename Dtype> template<index_t Dim1>
inline Tensor<Dim1, Dtype> Tensor<Dim, Dtype>::view(const Shape<Dim1>& shape) const {
    CHECK_TENSOR_CONTIGUOUS(*this, "view");
    CHECK_SIZE_EQUAL(shape_.dsize(), shape.dsize());
    return Tensor<Dim1, Dtype>(*this, shape);
}

template<index_t Dim, typename Dtype>
inline Dtype Tensor<Dim, Dtype>::eval(index_t* ids) const {
    int offset = 0;
    for(index_t i = 0; i < Dim; i++)
        offset += stride_[i] * ids[i];
    return storage_[offset];
}

template<index_t Dim, typename Dtype>
inline Dtype& Tensor<Dim, Dtype>::eval(index_t* ids) {
    int offset = 0;
    for(index_t i = 0; i < Dim; i++)
        offset += stride_[i] * ids[i];
    return storage_[offset];
}

template<index_t Dim, typename Dtype>
inline Tensor<Dim, Dtype>& Tensor<Dim, Dtype>::operator=(const Exp<Dtype>& src) {
    index_t loc[Dim];
    index_t idx = 0;
    for(int i = 0; i < Dim; i++) loc[i] = -1;

    while(idx >= 0)  {
        if(idx == Dim) {
            eval(loc) = src.eval(loc);
            idx --; 
        } else if(loc[idx] < shape_[idx] - 1) {
            loc[idx] ++;
            if(idx < Dim - 1) loc[idx+1] = -1;
            idx++;
        } else idx--;
    }
    return *this;
}

} // namespace el (general definition)


namespace el {
// specialized definition

template<typename Dtype>
std::ostream& operator<<(std::ostream& out, const Tensor<1, Dtype>& t) {
    out << '[';
    for(index_t i = 0; i < t.size(0); i++) {
        out << t[{i}];
        if(i != t.size(0) - 1) out << ", ";
    }
    out << ']';
}

template<typename Dtype>
std::ostream& operator<<(std::ostream& out, const Tensor<2, Dtype>& t) {
    out << '[';
    for(index_t i = 0; i < t.size(0); i++) {
        out << t.slice(i);
        if(i != t.size(0) - 1) out << ',' << '\n';
    }
    out << ']';
}

template<index_t Dim, typename Dtype>
std::ostream& operator<<(std::ostream& out, const Tensor<Dim, Dtype>& t) {
    out << '[';
    for(index_t i = 0; i < t.size(0); i++) {
        out << t.slice(i);
        if(i != t.size(0) - 1) out << "," << '\n' << '\n';
    }
    out << ']';
}

} // namespace el
#endif // TENSOR_TENSOR_IMPL_H_
