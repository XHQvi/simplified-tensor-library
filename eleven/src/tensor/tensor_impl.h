#ifndef TENSOR_TENSOR_IMPL_H_
#define TENSOR_TENSOR_IMPL_H_

#include <iostream>
#include <initializer_list>
#include "storage.h"
#include "shape.h"
#include "../expression/expression.h"
#include "tensor.h"

namespace el {

template<typename Dtype>
class Tensor: public Exp<Dtype> {
public:
    // constructor
    Tensor(const Storage<Dtype>& storage, const Shape& shape, bool requires_grad=false);
    Tensor(const Dtype* data, const Shape& shape, bool requires_grad=false);
    explicit Tensor(const Shape& shape, bool requires_grad=false);
    Tensor(const Tensor& other);
    ~Tensor();

    // method
    index_t dim(void) const;
    index_t size(index_t idx) const;
    index_t offset(void) const;
    const Shape& size(void) const;
    const IndexArray& stride(void) const;
    index_t version(void) const;
    Dtype& operator[](std::initializer_list<index_t> ids);
    const Dtype& operator[](std::initializer_list<index_t> ids) const;
    Tensor slice(index_t idx, index_t dim=0) const;
    Tensor slice(index_t start_idx, index_t end_idx, index_t dim) const;
    Tensor transpose(index_t dim1, index_t dim2) const;
    Tensor view(const Shape& shape) const;
    bool is_contiguous(void) const;
    Tensor& operator=(const Exp<Dtype>& src);
    Tensor& operator=(const Tensor& src);
    void backward(void) const;
    // friend
    template<typename Dtype1> friend std::ostream& operator<<(std::ostream& out, const Tensor<Dtype1>& t);
    friend Exp<Dtype>;
private:
    Storage<Dtype> storage_;
    Shape shape_;
    IndexArray stride_;

    // constructor
    Tensor(const Tensor& other, const Shape& shape);
    Tensor(const Storage<Dtype>& storage, const Shape& shape, const IndexArray& stride, bool requires_grad=false);
    // method for implementing template expression
    void set_self(const Exp<Dtype>& src);
    Dtype eval(index_t* ids) const;
    Dtype& eval(index_t* ids);
    
    // auto gradient
    struct AutoGradMeta {
        index_t ref_count = 1;
        Tensor<Dtype> grad_;
        const Exp<Dtype>& next_exp_;
        bool from_view_;
        AutoGradMeta(const Storage<Dtype>& storage, const Shape& shape, const IndexArray& stride, 
                     const Exp<Dtype>& next_exp, bool from_view);
        AutoGradMeta(const Storage<Dtype>& storage, const Shape& shape,
                     const Exp<Dtype>& next_exp, bool from_view);
        AutoGradMeta(const Shape& shape, const Exp<Dtype>& next_exp, bool from_view);
    };
    // It's annoying to use smate pointer here, so I maintain ref_count by myself.
    // Increase ref_count in copy constructor, and decrease it in deconstructor.
    // So never assign ag_meta to another.
    AutoGradMeta* ag_meta_ = nullptr;
    bool requires_grad_;
};

// ******************** constructors of AutoGradMeta ********************
template<typename Dtype>
Tensor<Dtype>::AutoGradMeta::AutoGradMeta(const Storage<Dtype>& storage, const Shape& shape, const IndexArray& stride, 
                                          const Exp<Dtype>& next_exp, bool from_view)
    : grad_(storage, shape, stride, false), next_exp_(next_exp), from_view_(from_view) {}

template<typename Dtype>
Tensor<Dtype>::AutoGradMeta::AutoGradMeta(const Storage<Dtype>& storage, const Shape& shape,
                                          const Exp<Dtype>& next_exp, bool from_view)
    : grad_(storage, shape, false), next_exp_(next_exp), from_view_(from_view) {}

template<typename Dtype>
Tensor<Dtype>::AutoGradMeta::AutoGradMeta(const Shape& shape, const Exp<Dtype>& next_exp, bool from_view)
    : AutoGradMeta(Storage<Dtype>(0, shape.dsize()), shape, next_exp, from_view) {}


// ******************** constructors of Tensor ********************
template<typename Dtype>
Tensor<Dtype>::Tensor(const Storage<Dtype>& storage, const Shape& shape, const IndexArray& stride, bool requires_grad)
    : storage_(storage), shape_(shape), stride_(stride), requires_grad_(requires_grad) {}

template<typename Dtype>
Tensor<Dtype>::Tensor(const Storage<Dtype>& storage, const Shape& shape, bool requires_grad)
    : storage_(storage), shape_(shape), stride_(shape_.dim()), requires_grad_(requires_grad) {
    for(index_t i = 0; i < shape_.dim(); i++) {
        if(shape_[i] == 1) stride_[i] = 0; // for broadcasting
        else stride_[i] = shape_.subsize(i + 1);
    }
}

template<typename Dtype>
Tensor<Dtype>::Tensor(const Dtype* data, const Shape& shape, bool requires_grad)
    : Tensor<Dtype>(Storage<Dtype>(data, shape.dsize()), shape, requires_grad) {}

template<typename Dtype>
Tensor<Dtype>::Tensor(const Shape& shape, bool requires_grad)
    : Tensor<Dtype>(Storage<Dtype>(shape.dsize()), shape, requires_grad) {}

template<typename Dtype>
Tensor<Dtype>::Tensor(const Tensor<Dtype>& other, const Shape& shape)
    : Tensor<Dtype>(other.storage_, shape, other.requires_grad_) {}

template<typename Dtype>
Tensor<Dtype>::Tensor(const Tensor<Dtype>& other)
    : storage_(other.storage_), shape_(other.shape_), stride_(other.stride_), 
      ag_meta_(other.ag_meta_), requires_grad_(other.requires_grad_) {
    if(ag_meta_) ag_meta_->ref_count ++;
}

template<typename Dtype>
Tensor<Dtype>::~Tensor() {
    if(ag_meta_) {
        ag_meta_->ref_count --;
        if(ag_meta_->ref_count == 0) delete ag_meta_;
    }
}

// ******************** Methods of Tensor ********************
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
inline index_t Tensor<Dtype>::version(void) const {return storage_.version();}

// This function will change the content of tensor, so version of the storage will be add 1.
// If the tensor has been in a computation graph, an exception would be throwed when gradient backwards.
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


// Function slice(), transpose() and view() will return new tensor which shares one storage with 
// the origin tensor. Correspondingly, their grad should also share one storage. And their grad's
// shape and stride shoule be the same as themselves. So in fact, grad can be just a storage omitting
// shape and stride information. But for convenience, I use Tensor instead of Storage as grad.
//
// It took me some time to figure out the relation between grad of view tensors(I mean, tensors returned by slice,
// transpose or view) and original tensors.
// 1. they should share the same storage.
// 2. view tensors' backward() should accumulate gradient according to their own shape and stride.
// 3. view tensors' backward() will call origin tensors' backward(), but doesn't pass any grad.
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
    
    Tensor<Dtype> ret(storage, shape, stride, requires_grad_);
    if(requires_grad_ && ag_meta_) {
        Storage<Dtype> grad_storage(ag_meta_->grad_.storage_, stride_[dim] * idx);
        ret.ag_meta_ = new AutoGradMeta(grad_storage, shape, stride, *this, true);
    }
    return ret;
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

    Tensor<Dtype> ret(storage, shape, stride, requires_grad_);
    if(requires_grad_ && ag_meta_) {
        Storage<Dtype> grad_storage(ag_meta_->grad_.storage_, stride_[dim] * start_idx);
        ret.ag_meta_ = new AutoGradMeta(grad_storage, shape, stride, *this, true);
    }
    return ret;
}

template<typename Dtype>
inline Tensor<Dtype> Tensor<Dtype>::transpose(index_t dim1, index_t dim2) const {
    CHECK_BETWEEN(dim1, 0, shape_.dim(), IndexOutOfRange,
        "%dD tensor got %d dimension index", shape_.dim(), dim1);
    CHECK_BETWEEN(dim2, 0, shape_.dim(), IndexOutOfRange,
        "%dD tensor got %d dimension index", shape_.dim(), dim2);

    Shape shape(shape_);
    shape[dim1] = shape_[dim2];
    shape[dim2] = shape_[dim1];

    IndexArray stride(stride_);
    stride[dim1] = stride_[dim2];
    stride[dim2] = stride_[dim1];
    
    Tensor<Dtype> ret(storage_, shape, stride, requires_grad_);
    if(requires_grad_ && ag_meta_)
        ret.ag_meta_ = new AutoGradMeta(ag_meta_->grad_.storage_, shape, stride, *this, true);
    return ret;
}

template<typename Dtype>
inline Tensor<Dtype> Tensor<Dtype>::view(const Shape& shape) const {
    CHECK_TRUE(is_contiguous(), TensorNotContiguous,
        "Tensor can't be viewed, which is not is_contiguous");
    CHECK_EQUAL(shape.dsize(), shape_.dsize(), DsizeNotMatch,
        "Got shape with dsize %d doesn't match original dsize %d", shape.dsize(), shape_.dsize());

    Tensor<Dtype> ret(*this, shape);
    if(requires_grad_ && ag_meta_)
        ret.ag_meta_ = new AutoGradMeta(ag_meta_->grad_.storage_, shape, *this, true);
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
Dtype Tensor<Dtype>::eval(index_t* ids) const {
    int offset = 0;
    for(index_t i = 0; i < shape_.dim(); i++)
        offset += stride_[i] * ids[i];
    return storage_[offset];
}

// This function will change the content of tensor, but won't change version of storage.
// It may cause wrong gradient. So this function is dangerous.
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

// This function will change the content of tensor, so version of the storage will be add 1.
// If the tensor has been in a computation graph, an exception would be throwed when gradient backwards.
//
// If requires_grad is true, alloc a new AutoGradMeta for this tensor no matter ag_meta is nullptr or not.
// That means this tensor would be detached from its original graph.
template<typename Dtype>
inline Tensor<Dtype>& Tensor<Dtype>::operator=(const Exp<Dtype>& src) {
    CHECK_BROADCAST(*this, src);
    set_self(src);
    if(requires_grad_) {
        delete ag_meta_;
        ag_meta_ = new AutoGradMeta(shape_, src, false);
    }
    return *this;
}

// This function is basically equal to the previous one.
template<typename Dtype>
inline Tensor<Dtype>& Tensor<Dtype>::operator=(const Tensor<Dtype>& src) {
    CHECK_BROADCAST(*this, src);
    set_self(src);
    if(requires_grad_) {
        delete ag_meta_;
        ag_meta_ = new AutoGradMeta(shape_, src, false);
    }
    return *this;
}

template<typename Dtype>
inline void Tensor<Dtype>::backward(void) const {
    std::cout << "tensor backward" << std::endl;
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
