#ifndef TENSOR_TENSOR_IMPL_H_
#define TENSOR_TENSOR_IMPL_H_

#include <iostream>
#include <initializer_list>
#include "storage.h"
#include "shape.h"
#include "../expression/expression.h"
#include "../expression/node.h"
#include "tensor.h"

namespace el {

template<typename Dtype>
class Tensor: public Exp<Dtype> {
public:
    // constructor
    Tensor(const Storage<Dtype>& storage, const Shape& shape, bool requires_grad=false);
    Tensor(const Dtype* data, const Shape& shape, bool requires_grad=false);
    explicit Tensor(const Shape& shape, bool requires_grad=false);
    Tensor(const Tensor& other) = default;


    index_t dim(void) const;
    index_t size(index_t idx) const;
    index_t offset(void) const;
    const Shape& size(void) const;
    const IndexArray& stride(void) const;
    index_t version(void) const;
    bool requires_grad(void) const;

    // operator[] can modify data of a tensor, and by the same time will increment version of the tensor.
    // Like in-place operation to a tensor in Pytorch, this operation may damage the computation graph,
    // which means aee
    Dtype& operator[](std::initializer_list<index_t> ids);
    const Dtype& operator[](std::initializer_list<index_t> ids) const;
    
    // Methods with a underscore as suffix will return a pointer which points to a tensor dynamically allocated.
    // The other ones will return a tensor object on stack storage. 
    // But all tensors, whether are static allocated or dynamic allocated, share the same storage space with 
    // the original tensor.
    Tensor slice(index_t idx, index_t dim=0) const;
    Tensor slice(index_t start_idx, index_t end_idx, index_t dim) const;
    Tensor transpose(index_t dim1, index_t dim2) const;
    Tensor view(const Shape& shape) const;
    Tensor squeeze(void) const;
    Tensor unsqueeze(index_t dim) const;
    Tensor* slice_(index_t idx, index_t dim=0) const;
    Tensor* slice_(index_t start_idx, index_t end_idx, index_t dim) const;
    Tensor* transpose_(index_t dim1, index_t dim2) const;
    Tensor* view_(const Shape& shape) const;
    Tensor* squeeze_(void) const;
    Tensor* unsqueeze_(index_t dim) const;
    bool is_contiguous(void) const;

    // Assigning a tensor to Exp or Tensor, won't add the tensor to any computation graphs, but to a node will.
    Tensor& operator=(const Exp<Dtype>& src);
    Tensor& operator=(const Tensor& src);
    Tensor& operator=(const Node<Dtype>& src);
    Tensor& operator+=(const Exp<Dtype>& src);
    void backward(const Exp<Dtype>& grad) const;
    const Tensor& grad(void) const;
    // These functions can access and modify data bypassing inspections, and they won't increment the version 
    // of this tensor. So using these function to a tensor in a computation graph may cause concealed gradient 
    // calculation error.
    Dtype eval(index_t* ids) const;
    Dtype& eval(index_t* ids);
    Dtype eval(index_t idx) const;
    Dtype& eval(index_t idx);

    template<typename Dtype1> friend std::ostream& operator<<(std::ostream& out, const Tensor<Dtype1>& t);
private:
    Storage<Dtype> storage_;
    Shape shape_;
    IndexArray stride_;

    // auto gradient
    struct AutoGradMeta {
        Tensor<Dtype> grad_;
        bool from_view_;
        ConstExptr<Dtype> next_exp_;
        AutoGradMeta(const Storage<Dtype>& storage, const Shape& shape, const IndexArray& stride, 
                     const Exp<Dtype>* next_exp, bool from_view);
        AutoGradMeta(const Storage<Dtype>& storage, const Shape& shape,
                     const Exp<Dtype>* next_exp, bool from_view);
        AutoGradMeta(const Shape& shape);
    };
    std::shared_ptr<AutoGradMeta> ag_meta_;
    bool requires_grad_;

    // constructor
    Tensor(const Storage<Dtype>& storage, const Shape& shape, const IndexArray& stride, bool requires_grad=false);
    // methods
    void set_self(const Exp<Dtype>& src);
};

// ******************** constructors and methods of AutoGradMeta ********************
template<typename Dtype>
Tensor<Dtype>::AutoGradMeta::AutoGradMeta(const Storage<Dtype>& storage, const Shape& shape, const IndexArray& stride, 
                                          const Exp<Dtype>* next_exp, bool from_view)
    : grad_(storage, shape, stride, false), next_exp_(next_exp, true), from_view_(from_view) {
    ConstExptr<Dtype>::make_uncontrol(grad_);
}

template<typename Dtype>
Tensor<Dtype>::AutoGradMeta::AutoGradMeta(const Storage<Dtype>& storage, const Shape& shape,
                                          const Exp<Dtype>* next_exp, bool from_view)
    : grad_(storage, shape, false), next_exp_(next_exp, true), from_view_(from_view) {
    ConstExptr<Dtype>::make_uncontrol(grad_);   
}

template<typename Dtype>
Tensor<Dtype>::AutoGradMeta::AutoGradMeta(const Shape& shape)
    : grad_(Storage<Dtype>(shape.dsize(), 0), shape, false), next_exp_(), from_view_(false) {
    ConstExptr<Dtype>::make_uncontrol(grad_);
}


// ******************** constructors of Tensor ********************
template<typename Dtype>
Tensor<Dtype>::Tensor(const Storage<Dtype>& storage, const Shape& shape, const IndexArray& stride, bool requires_grad)
    : storage_(storage), shape_(shape), stride_(stride), requires_grad_(requires_grad) {
    if(requires_grad_) 
        ag_meta_.reset(new AutoGradMeta(shape_));
}

template<typename Dtype>
Tensor<Dtype>::Tensor(const Storage<Dtype>& storage, const Shape& shape, bool requires_grad)
    : storage_(storage), shape_(shape), stride_(shape_.dim()), requires_grad_(requires_grad) {
    for(index_t i = 0; i < shape_.dim(); i++) {
        if(shape_[i] == 1) stride_[i] = 0; // for broadcasting
        else stride_[i] = shape_.subsize(i + 1);
    }
    if(requires_grad_)
        ag_meta_.reset(new AutoGradMeta(shape_));
}

template<typename Dtype>
Tensor<Dtype>::Tensor(const Dtype* data, const Shape& shape, bool requires_grad)
    : Tensor<Dtype>(Storage<Dtype>(data, shape.dsize()), shape, requires_grad) {}

template<typename Dtype>
Tensor<Dtype>::Tensor(const Shape& shape, bool requires_grad)
    : Tensor<Dtype>(Storage<Dtype>(shape.dsize()), shape, requires_grad) {}


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

template<typename Dtype>
inline bool Tensor<Dtype>::requires_grad(void) const {return requires_grad_;}

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
    
    // requires_grad = false, to avoid creating extra AutoGradMeta
    Tensor<Dtype> ret(storage, shape, stride, false);
    if(requires_grad_) {
        ret.requires_grad_ = true;
        Storage<Dtype> grad_storage(ag_meta_->grad_.storage_, stride_[dim] * idx);
        ret.ag_meta_.reset(new AutoGradMeta(grad_storage, shape, stride, this, true));
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

    Tensor<Dtype> ret(storage, shape, stride, false);
    if(requires_grad_) {
        ret.requires_grad_ = true;
        Storage<Dtype> grad_storage(ag_meta_->grad_.storage_, stride_[dim] * start_idx);
        ret.ag_meta_.reset(new AutoGradMeta(grad_storage, shape, stride, this, true));
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
    
    Tensor<Dtype> ret(storage_, shape, stride, false);
    if(requires_grad_) {
        ret.requires_grad_ = true;
        ret.ag_meta_.reset(new AutoGradMeta(ag_meta_->grad_.storage_, shape, stride, this, true));
    }
    return ret;
}

template<typename Dtype>
inline Tensor<Dtype> Tensor<Dtype>::view(const Shape& shape) const {
    CHECK_TRUE(is_contiguous(), TensorNotContiguous,
        "Tensor can't be viewed, which is not is_contiguous");
    CHECK_EQUAL(shape.dsize(), shape_.dsize(), DsizeNotMatch,
        "Got shape with dsize %d doesn't match original dsize %d", shape.dsize(), shape_.dsize());

    Tensor<Dtype> ret(storage_, shape, false);
    if(requires_grad_) {
        ret.requires_grad_ = true;
        ret.ag_meta_.reset(new AutoGradMeta(ag_meta_->grad_.storage_, shape, this, true));
    }
    return ret;
}

template<typename Dtype>
inline Tensor<Dtype> Tensor<Dtype>::squeeze(void) const {
    index_t count = 0;
    index_t *dims = new int[shape_.dim()];

    for(index_t i = 0; i < shape_.dim(); i++)
        if(shape_[i] != 1)
            dims[count++] = shape_[i];
    Shape squeeze_shape(dims, count);
    delete [] dims;
    return view(squeeze_shape);
}

template<typename Dtype>
inline Tensor<Dtype> Tensor<Dtype>::unsqueeze(index_t dim) const {
    CHECK_BETWEEN(dim, 0, shape_.dim() + 1, IndexOutOfRange,
        "%dD Tensor can be unsqueezed on [0, %d] dimensions, but got dimension %d.", 
        shape_.dim(), shape_.dim(), dim);

    Shape unsqueeze_shape(nullptr, shape_.dim() + 1);
    index_t i = 0;
    for(; i != dim; i++)
        unsqueeze_shape[i] = shape_[i];
    unsqueeze_shape[dim] = 1;
    for(i++; i < unsqueeze_shape.dim(); i++)
        unsqueeze_shape[i] = shape_[i-1];
    return view(unsqueeze_shape);
}

template<typename Dtype>
Tensor<Dtype>* Tensor<Dtype>::slice_(index_t idx, index_t dim) const {
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
    
    // requires_grad = false, to avoid creating extra AutoGradMeta
    Tensor<Dtype>* ret = new Tensor<Dtype>(storage, shape, stride, false);
    if(requires_grad_) {
        ret->requires_grad_ = true;
        Storage<Dtype> grad_storage(ag_meta_->grad_.storage_, stride_[dim] * idx);
        ret->ag_meta_.reset(new AutoGradMeta(grad_storage, shape, stride, this, true));
    }
    return ret;
}

template<typename Dtype>
inline Tensor<Dtype>* Tensor<Dtype>::slice_(index_t start_idx, index_t end_idx, index_t dim) const {
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

    Tensor<Dtype>* ret = new Tensor<Dtype>(storage, shape, stride, false);
    if(requires_grad_) {
        ret->requires_grad_ = true;
        Storage<Dtype> grad_storage(ag_meta_->grad_.storage_, stride_[dim] * start_idx);
        ret->ag_meta_.reset(new AutoGradMeta(grad_storage, shape, stride, this, true));
    }
    return ret;
}

template<typename Dtype>
inline Tensor<Dtype>* Tensor<Dtype>::transpose_(index_t dim1, index_t dim2) const {
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
    
    Tensor<Dtype>* ret = new Tensor<Dtype>(storage_, shape, stride, false);
    if(requires_grad_) {
        ret->requires_grad_ = true;
        ret->ag_meta_.reset(new AutoGradMeta(ag_meta_->grad_.storage_, shape, stride, this, true));
    }
    return ret;
}

template<typename Dtype>
inline Tensor<Dtype>* Tensor<Dtype>::view_(const Shape& shape) const {
    CHECK_TRUE(is_contiguous(), TensorNotContiguous,
        "The tensor can't be viewed, which is not is_contiguous");
    CHECK_EQUAL(shape.dsize(), shape_.dsize(), DsizeNotMatch,
        "Got shape with dsize %d doesn't match original dsize %d", shape.dsize(), shape_.dsize());

    Tensor<Dtype>* ret = new Tensor<Dtype>(storage_, shape, false);
    if(requires_grad_) {
        ret->requires_grad_ = true;
        ret->ag_meta_.reset(new AutoGradMeta(ag_meta_->grad_.storage_, shape, this, true));
    }
    return ret;
}

template<typename Dtype>
inline Tensor<Dtype>* Tensor<Dtype>::squeeze_(void) const {
    index_t count = 0;
    index_t *dims = new int[shape_.dim()];

    for(index_t i = 0; i < shape_.dim(); i++)
        if(shape_[i] != 1)
            dims[count++] = shape_[i];
    Shape squeeze_shape(dims, count);
    delete [] dims;
    return view_(squeeze_shape);
}

template<typename Dtype>
inline Tensor<Dtype>* Tensor<Dtype>::unsqueeze_(index_t dim) const {
    CHECK_BETWEEN(dim, 0, shape_.dim() + 1, IndexOutOfRange,
        "%dD Tensor can be unsqueezed on [0, %d] dimensions, but got dimension %d.", 
        shape_.dim(), shape_.dim(), dim);

    Shape unsqueeze_shape(nullptr, shape_.dim() + 1);
    index_t i = 0;
    for(; i != dim; i++)
        unsqueeze_shape[i] = shape_[i];
    unsqueeze_shape[dim] = 1;
    for(i++; i < unsqueeze_shape.dim(); i++)
        unsqueeze_shape[i] = shape_[i-1];
    return view_(unsqueeze_shape);
}

template<typename Dtype>
bool Tensor<Dtype>::is_contiguous(void) const {
    for(index_t i = 0; i < shape_.dim(); i++)
        if(stride_[i] != 0 && stride_[i] != shape_.subsize(i+1))
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
// It may cause wrong gradient. Using it cautiously.
template<typename Dtype>
Dtype& Tensor<Dtype>::eval(index_t* ids) {
    int offset = 0;
    for(index_t i = 0; i < shape_.dim(); i++)
        offset += stride_[i] * ids[i];
    return storage_[offset];
}

template<typename Dtype>
Dtype Tensor<Dtype>::eval(index_t idx) const {return storage_[idx];}

template<typename Dtype>
Dtype& Tensor<Dtype>::eval(index_t idx) {return storage_[idx];}

// This function was written in a recursive form originally, then was converted to a while loop form.
// The while loop will iterate all possible indice for this tensor, so we can calculate and set each
// value in this tensor. For element-wise operation, it's fine to use a loop like 
//  for(index i = 0; i < storage_.size(); i++)
//      storage_[i] = calculate_value(i);
// But if we want to implement other operations, like Matrix Multiply and 2D Convolution, in the single
// function, we need logical indice instead of a physical index.
template<typename Dtype>
void Tensor<Dtype>::set_self(const Exp<Dtype>& src) {
    index_t num_dim = shape_.dim();
    index_t* loc = new index_t[num_dim];
    index_t idx = 0;
    IndexArray shape(num_dim);
    for(int i = 0; i < num_dim; i++) {
        loc[i] = -1;
        shape[i] = std::max(shape_[i], src.size(i));  // broadcasting
    }

    while(idx >= 0)  {
        if(idx == num_dim) {
            eval(loc) = src.eval(loc);
            idx --; 
        } else if(loc[idx] < shape[idx] - 1) {
            loc[idx] ++;
            if(idx < num_dim - 1) loc[idx+1] = -1;
            idx++;
        } else idx--;
    }
    delete [] loc;
    storage_.version_forward();
}

// This function will change the content of tensor, so version of the storage will be add 1.
// If the tensor has been in a computation graph, an exception would be thrown when gradient backwards.
//
// Assigning a Exp to a tensor wouldn't be added into any computation graphs, while assigning a node to a 
// tensor would. Because we couldn't assign ag_meta->next_exp_ to a exp when we only get the exp itself. 
// ag_meta->next_exp_ is a shared_ptr, and Exp has no method like shared_from_this.
// And it's unnecessary to implement that, considering we can use Node when we want to calculate gradient.
template<typename Dtype>
inline Tensor<Dtype>& Tensor<Dtype>::operator=(const Exp<Dtype>& src) {
    CHECK_BROADCAST(*this, src);
    set_self(src);
    return *this;
}

// Using src.shared_self(), we can add this operation into a computation graph, but I didn't do 
// that, considering of consistency.
template<typename Dtype>
inline Tensor<Dtype>& Tensor<Dtype>::operator=(const Tensor<Dtype>& src) {
    CHECK_BROADCAST(*this, src);
    set_self(src);
    return *this;
}

template<typename Dtype>
inline Tensor<Dtype>& Tensor<Dtype>::operator=(const Node<Dtype>& src) {
    const Exp<Dtype>& src_exp = src.get_exp();
    CHECK_BROADCAST(*this, src_exp);
    set_self(src_exp);
    if(requires_grad_)
        ag_meta_->next_exp_.reset(src.get_exp_ptr(), true);
    return *this;
}

template<typename Dtype>
inline Tensor<Dtype>& Tensor<Dtype>::operator+=(const Exp<Dtype>& src) {
    CHECK_BROADCAST(*this, src);
    index_t num_dim = shape_.dim();
    index_t* loc = new index_t[num_dim];
    index_t idx = 0;
    IndexArray shape(num_dim);
    for(int i = 0; i < num_dim; i++) {
        loc[i] = -1;
        shape[i] = std::max(shape_[i], src.size(i));
    }

    while(idx >= 0)  {
        if(idx == num_dim) {
            eval(loc) += src.eval(loc);
            idx --; 
        } else if(loc[idx] < shape[idx] - 1) {
            loc[idx] ++;
            if(idx < num_dim - 1) loc[idx+1] = -1;
            idx++;
        } else idx--;
    }
    delete [] loc;
    storage_.version_forward();
}

template<typename Dtype>
inline void Tensor<Dtype>::backward(const Exp<Dtype>& grad) const {
    CHECK_TRUE(requires_grad_, TensorNoGrad,
        "Call backward for a tensor with requires_grad false");

    ag_meta_->grad_ += grad;
    if(ag_meta_->next_exp_ && ConstExptr<Dtype>::grad_ready(*this)) {
        if(ag_meta_->from_view_) {
            index_t dsize = ag_meta_->next_exp_->dim();
            Storage<Dtype> storage(dsize, 0);
            Tensor<Dtype> view_grad(storage, Shape(nullptr, dsize), false);
            ConstExptr<Dtype>::make_uncontrol(view_grad);
            ag_meta_->next_exp_.backward(view_grad);
        } else {
            ag_meta_->next_exp_.backward(ag_meta_->grad_);
        }
    }
}

template<typename Dtype>
inline const Tensor<Dtype>& Tensor<Dtype>::grad(void) const {
    CHECK_TRUE(requires_grad_, TensorNoGrad,
        "Call grad() on a tensor with requires_grad false.");
    return ag_meta_->grad_;
}

template<typename Dtype>
std::ostream& operator<<(std::ostream& out, const Tensor<Dtype>& src) {
    Tensor<Dtype> t(src);
    t.requires_grad_ = false;

    std::ios_base::fmtflags flags = out.flags();
    out.setf(std::ios::fixed);
    out.precision(4);

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

    out.setf(flags);
    return out;
}

} // namespace el
#endif // TENSOR_TENSOR_IMPL_H_
