#ifndef TENSOR_TENSOR_IMPL_H_
#define TENSOR_TENSOR_IMPL_H_

#include <iostream>
#include <initializer_list>
#include "storage.h"
#include "shape.h"
#include "../expression/expression.h"

namespace el {
// general declaration
template<typename Dtype>
class Tensor: public Exp<Dtype> {
public:
    // constructor
    Tensor(const Storage<Dtype>& storage, const Shape& shape);
    Tensor(const Dtype* data, const Shape& shape);
    explicit Tensor(const Shape& shape);
    Tensor(const Tensor& other) = default;

    // method
    index_t dim(void) const;
    index_t size(index_t idx) const;
    index_t offset(void) const;
    const Shape& size(void) const;
    const IndexArray& stride(void) const;
    bool is_unchanged(void) const;
    Dtype& operator[](std::initializer_list<index_t> ids);
    const Dtype& operator[](std::initializer_list<index_t> ids) const;
    Tensor slice(index_t idx, index_t dim=0) const;
    Tensor slice(index_t start_idx, index_t end_idx, index_t dim) const;
    Tensor transpose(index_t dim1, index_t dim2) const;
    bool is_contiguous(void) const;
    Tensor view(const Shape& shape) const;
    Tensor& operator=(const Exp<Dtype>& src);
    Tensor& operator=(const Tensor& src);
    // friend
    template<typename Dtype1> friend std::ostream& operator<<(std::ostream& out, const Tensor<Dtype1>& t);
    friend Exp<Dtype>;
private:
    Storage<Dtype> storage_;
    Shape shape_;
    IndexArray stride_;
    const index_t version_;

    Tensor(const Tensor& other, const Shape& shape);
    Tensor(const Storage<Dtype>& storage, const Shape& shape, const IndexArray& stride);
    // method for implementing template expression
    void set_self(const Exp<Dtype>& src);
    Dtype eval(index_t* ids) const;
    Dtype& eval(index_t* ids);
};
} // namespace el (general declaration)

namespace el {

template<typename Dtype>
Tensor<Dtype>::Tensor(const Storage<Dtype>& storage, const Shape& shape, const IndexArray& stride)
    : storage_(storage), shape_(shape), stride_(stride), version_(storage_.version()) {}

template<typename Dtype>
Tensor<Dtype>::Tensor(const Storage<Dtype>& storage, const Shape& shape)
    : storage_(storage), shape_(shape), stride_(shape_.dim()), version_(storage_.version()) {
    for(index_t i = 0; i < shape_.dim(); i++) {
        if(shape_[i] == 1) stride_[i] = 0; // for broadcasting
        else stride_[i] = shape_.subsize(i + 1);
    }
}

template<typename Dtype>
Tensor<Dtype>::Tensor(const Dtype* data, const Shape& shape): Tensor<Dtype>(Storage<Dtype>(data, shape.dsize()), shape){}

template<typename Dtype>
Tensor<Dtype>::Tensor(const Shape& shape): Tensor<Dtype>(Storage<Dtype>(shape.dsize()), shape) {}

template<typename Dtype>
Tensor<Dtype>::Tensor(const Tensor<Dtype>& other, const Shape& shape): Tensor<Dtype>(other.storage_, shape) {}

template<typename Dtype>
inline index_t Tensor<Dtype>::dim(void) const {return shape_.dim();}

template<typename Dtype>
 inline index_t Tensor<Dtype>::offset(void) const {return storage_.offset();}

template<typename Dtype>
inline index_t Tensor<Dtype>::size(index_t idx) const {
    CHECK_BETWEEN(idx, 0, shape_.dim(), IndexOutOfRange,
       "%dD tensor got %d dimension index", shape_.dim(), idx);
    return shape_[idx];
}

template<typename Dtype>
inline const Shape& Tensor<Dtype>::size(void) const {return shape_;}

template<typename Dtype>
inline const IndexArray& Tensor<Dtype>::stride(void) const {return stride_;}

template<typename Dtype>
inline bool Tensor<Dtype>::is_unchanged(void) const {return version_ == storage_.version();}

// This function will change the content of tensor, so version of the storage will be add 1,
// which causes failure of gradient backward.
template<typename Dtype>
Dtype& Tensor<Dtype>::operator[](std::initializer_list<index_t> ids) {
    CHECK_EQUAL(dim(), ids.size(), DimNotMatch,
        "%dD tensor got %dD indice", dim(), ids.size());

    index_t offset = 0, i = 0;
    for(auto idx: ids) {
        CHECK_BETWEEN(idx, 0, shape_[i], IndexOutOfRange,
            "Tensor has size %d on %d dimension, but got %d index", shape_[i], i, idx);
        offset += idx * stride_[i++];
    }
    storage_.version_forward();
    return storage_[offset];
}

template<typename Dtype>
const Dtype& Tensor<Dtype>::operator[](std::initializer_list<index_t> ids) const {
    CHECK_EQUAL(dim(), ids.size(), DimNotMatch,
        "%dD tensor got %dD indice", dim(), ids.size());

    index_t offset = 0, i = 0;
    for(auto idx: ids) {
        CHECK_BETWEEN(idx, 0, shape_[i], IndexOutOfRange,
            "Tensor has size %d on %d dimension, but got %d index", shape_[i], i, idx);
        offset += idx * stride_[i++];
    }
    return storage_[offset]; 
}

template<typename Dtype>
Tensor<Dtype> Tensor<Dtype>::slice(index_t idx, index_t dim) const {
    CHECK_BETWEEN(dim, 0, shape_.dim(), IndexOutOfRange,
        "%dD tensor got index on th%d dimension", shape_.dim(), idx);
    CHECK_BETWEEN(idx, 0, shape_[dim], IndexOutOfRange,
        "Tensor has size %d on %d dimension, but got index %d", shape_[dim], dim, idx);

    Storage<Dtype> storage(storage_, stride_[dim] * idx);
    Shape shape(shape_, dim);
    IndexArray stride(shape_.dim() - 1);

    int i = 0;
    for(; i != dim && i < shape_.dim()-1; i++)
        stride[i] = stride_[i];
    for(;i < shape_.dim()-1; i++)
        stride[i] = stride_[i+1];
    return Tensor<Dtype>(storage, shape, stride);
}

template<typename Dtype>
inline Tensor<Dtype> Tensor<Dtype>::slice(index_t start_idx, index_t end_idx, index_t dim) const {
    CHECK_BETWEEN(dim, 0, shape_.dim(), IndexOutOfRange,
        "%dD tensor got index on th%d dimension", shape_.dim(), dim);
    CHECK_BETWEEN(start_idx, 0, shape_[dim], IndexOutOfRange,
        "Tensor has size %d on %d dimension, but got %d index", shape_[dim], dim, start_idx);
    CHECK_BETWEEN(end_idx, 0, shape_[dim], IndexOutOfRange,
        "Tensor has size %d on %d dimension, but got %d index", shape_[dim], dim, end_idx);

    Storage<Dtype> storage(storage_, stride_[dim] * start_idx);
    Shape shape(shape_);
    IndexArray stride(stride_);
    shape[dim] = end_idx - start_idx;
    return Tensor<Dtype>(storage, shape, stride);
}

template<typename Dtype>
inline Tensor<Dtype> Tensor<Dtype>::transpose(index_t dim1, index_t dim2) const {
    CHECK_BETWEEN(dim1, 0, shape_.dim(), IndexOutOfRange,
        "%dD tensor got %d dimension index", shape_.dim(), dim1);
    CHECK_BETWEEN(dim2, 0, shape_.dim(), IndexOutOfRange,
        "%dD tensor got %d dimension index", shape_.dim(), dim2);

    Tensor<Dtype> ret(*this);
    index_t temp = ret.stride_[dim1];
    ret.stride_[dim1] = ret.stride_[dim2];
    ret.stride_[dim2] = temp;

    temp = ret.shape_[dim1];
    ret.shape_[dim1] = ret.shape_[dim2];
    ret.shape_[dim2] = temp;
    return ret;    
}

template<typename Dtype>
bool Tensor<Dtype>::is_contiguous(void) const {
    for(index_t i = 1; i <= shape_.dim(); i++)
        if(stride_[i-1] != shape_.subsize(i))
            return false;
    return true;
}

template<typename Dtype>
inline Tensor<Dtype> Tensor<Dtype>::view(const Shape& shape) const {
    CHECK_TRUE(is_contiguous(), TensorNotContiguous,
        "Tensor can't be viewed, which is not is_contiguous");
    CHECK_EQUAL(shape.dsize(), shape_.dsize(), DsizeNotMatch,
        "Got shape with dsize %d doesn't match original dsize %d", shape.dsize(), shape_.dsize());

    return Tensor<Dtype>(*this, shape);
}

template<typename Dtype>
Dtype Tensor<Dtype>::eval(index_t* ids) const {
    int offset = 0;
    for(index_t i = 0; i < shape_.dim(); i++)
        offset += stride_[i] * ids[i];
    return storage_[offset];
}

// This function will change the content of tensor, but won't change version of storage.
// It may cause wrong gradient. So this function is dangerous
template<typename Dtype>
Dtype& Tensor<Dtype>::eval(index_t* ids) {
    int offset = 0;
    for(index_t i = 0; i < shape_.dim(); i++)
        offset += stride_[i] * ids[i];
    return storage_[offset];
}

// This function was written in a recursive form originally, then was converted to a while loop form.
// The while loop will iterate all possible indice for this tensor, so we can calculate and set each
// value. For element-wise operation, it's fine to use a loop like 
//  for(index i = 0; i < storage_.size(); i++)
//      storage_[i] = calculate_value(i);
// But if we want to implement other operations, like Matrix Multiply and 2D Convolution, in the single
// function. So we need logical indice instead of a physical index.
template<typename Dtype>
void Tensor<Dtype>::set_self(const Exp<Dtype>& src) {
    index_t num_dim = shape_.dim();
    index_t* loc = new index_t[num_dim];
    index_t idx = 0;
    for(int i = 0; i < num_dim; i++) loc[i] = -1;

    while(idx >= 0)  {
        if(idx == num_dim) {
            eval(loc) = src.eval(loc);
            idx --; 
        } else if(loc[idx] < shape_[idx] - 1) {
            loc[idx] ++;
            if(idx < num_dim - 1) loc[idx+1] = -1;
            idx++;
        } else idx--;
    }
    delete [] loc;
    storage_.version_forward();
}

// This function will change the content of tensor, so version of the storage will be add 1,
// which causes failure of gradient backward.
template<typename Dtype>
inline Tensor<Dtype>& Tensor<Dtype>::operator=(const Exp<Dtype>& src) {
    CHECK_BROADCAST(*this, src);
    set_self(src);
    return *this;
}

// This function will change the content of tensor, so version of the storage will be add 1,
// which causes failure of gradient backward.
template<typename Dtype>
inline Tensor<Dtype>& Tensor<Dtype>::operator=(const Tensor<Dtype>& src) {
    CHECK_BROADCAST(*this, src);
    set_self(src);
    return *this;
}

template<typename Dtype>
std::ostream& operator<<(std::ostream& out, const Tensor<Dtype>& t) {
    out << '[';
    if(t.dim() == 1) {
        for(index_t i = 0; i < t.size(0); i++) {
            out << t[{i}];
            if(i != t.size(0) - 1) out << ", ";
        } 
    } else if(t.dim() == 2) {
        for(index_t i = 0; i < t.size(0); i++) {
            out << t.slice(i);
            if(i != t.size(0) - 1) out << ',' << '\n';
        }
    } else {
        for(index_t i = 0; i < t.size(0); i++) {
            out << t.slice(i);
            if(i != t.size(0) - 1) out << "," << '\n' << '\n';
        }
    }
    out << ']';
    return out;
}

} // namespace el
#endif // TENSOR_TENSOR_IMPL_H_
