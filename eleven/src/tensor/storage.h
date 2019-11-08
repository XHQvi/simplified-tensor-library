#ifndef TENSOR_STORAGE_H_
#define TENSOR_STORAGE_H_

#include<cstring>
#include "../utils/base.h"

namespace el {

template<typename Dtype>
class Storage {
    std::shared_ptr<char> bptr_;  // base pointer
    Dtype* dptr_;  // data pointer
    void init_version(void) {*reinterpret_cast<index_t*>(bptr_.get()) = 0;}
public:
    // constructor
    explicit Storage(index_t dsize)
        : bptr_(new char[dsize * sizeof(Dtype) + sizeof(index_t)]), 
          dptr_(reinterpret_cast<Dtype*>(bptr_.get() + sizeof(index_t))) {
        init_version();
    }
    Storage(const Storage& other, index_t offset)
        : bptr_(other.bptr_), 
          dptr_(other.dptr_ + offset){
        init_version();
    }
    explicit Storage(const Storage& other) = default;
    Storage(const Dtype* data, index_t dsize): Storage(dsize) {
        memcpy(dptr_, data, dsize*sizeof(Dtype));
    }
    Storage(int value, index_t dsize): Storage(dsize) {
        memset(dptr_, value, dsize*sizeof(Dtype));
    }
    // method
    const Dtype& operator[](index_t i) const {return dptr_[i];}
    Dtype& operator[](index_t i) {return dptr_[i];}
    index_t offset(void) const {return dptr_ - (Dtype*)(bptr_.get()) + sizeof(index_t);}
    index_t version(void) const {return *reinterpret_cast<index_t*>(bptr_.get());}
    void version_forward(void) const {*reinterpret_cast<index_t*>(bptr_.get()) += 1;}
    // ban
    Storage(void) = delete;
    Storage& operator=(const Storage& other) = delete;
    Storage(Storage&& other) = delete;
};

}  // namespace el
#endif
