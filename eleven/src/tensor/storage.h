#ifndef TENSOR_STORAGE_H_
#define TENSOR_STORAGE_H_

#include<memory>
#include "../base/type.h"

namespace el {

template<typename Dtype>
struct Storage {
    std::shared_ptr<Dtype> bptr_;  // base pointer
    Dtype* dptr_;  // data pointer

    // constructor
    explicit Storage(index_t dsize):bptr_(new Dtype[dsize]), dptr_(bptr_.get()){}
    Storage(const Storage& other, index_t offset): bptr_(other.bptr_), dptr_(other.dptr_ + offset){}
    explicit Storage(const Storage& other) = default;
    Storage(const Dtype* data, index_t dsize): Storage(dsize) {
        memcpy(dptr_, data, dsize*sizeof(Dtype));
    }
    // method
    const Dtype& operator[](index_t i) const {return dptr_[i];}
    Dtype& operator[](index_t i) {return dptr_[i];}
    index_t offset(void) const {return dptr_ - bptr_.get();}
    // ban
    Storage(void) = delete;
    Storage& operator=(const Storage& other) = delete;
    Storage(Storage&& other) = delete;
};

}  // namespace el
#endif
