#ifndef UTILS_FIXED_SIZE_ARRAY_H_
#define UTILS_FIXED_SIZE_ARRAY_H_

#include <iostream>
#include <memory>
#include <initializer_list>

namespace el {

template<typename Dtype>
class FixedSizeArray {
public:
	FixedSizeArray() = default;
	explicit FixedSizeArray(std::initializer_list<Dtype> data);
	explicit FixedSizeArray(size_t size);
	explicit FixedSizeArray(const FixedSizeArray& other);

	FixedSizeArray& operator=(const FixedSizeArray& other) = delete;

	void set(std::initializer_list<Dtype> data);
	size_t size(void) const {return size_;}
	Dtype& operator[](size_t i) {return dptr_.get()[i];}
	const Dtype& operator[](size_t i) const {return dptr_.get()[i];}

	template<typename Dtype1> friend std::ostream& operator<<(std::ostream& out, const FixedSizeArray<Dtype1>& fsarr);
private:
	const size_t size_;
	std::unique_ptr<Dtype[]> dptr_;
};

template<typename Dtype>
FixedSizeArray<Dtype>::FixedSizeArray(size_t size)
	: size_(size), dptr_(new Dtype[size_]) {}

template<typename Dtype>
FixedSizeArray<Dtype>::FixedSizeArray(std::initializer_list<Dtype> data)
	: FixedSizeArray(data.size()) {
	int i = 0;
	Dtype* ptr = dptr_.get();
	for(auto d: data) 
		ptr[i++] = d;
}

template<typename Dtype>
FixedSizeArray<Dtype>::FixedSizeArray(const FixedSizeArray& other)
	:FixedSizeArray(other.size()) {
	Dtype* ptr = dptr_.get();
	for(int i = 0; i < other.size(); i++) 
		ptr[i] = other[i];
}

template<typename Dtype>
void FixedSizeArray<Dtype>::set(std::initializer_list<Dtype> data) {
	size_ = data.size();
	dptr_ = new Dtype[size_];

	int i = 0;
	Dtype* ptr = dptr_.get();
	for(auto d: data) 
		ptr[i++] = d;
}

template<typename Dtype>
std::ostream& operator<<(std::ostream& out, const FixedSizeArray<Dtype>& fsarr) {
	out << '(' << fsarr[0];
	for(int i = 1; i < fsarr.size_; i++)
		out << ", " << fsarr[i];
	out << ')';
	return out;
}

}
#endif