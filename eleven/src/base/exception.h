#ifndef BASE_EXCEPTION_H_
#define BASE_EXCEPTION_H_

#include <exception>
#include <cstdio>
#include "type.h"

namespace el {
namespace exc {

enum ExcType : int{
	IndexOutOfRange_t = 0, DimNotMatch_t, SizeNotEqual_t, DimNotExist_t, OperatorNotBroadcast_t,
	ConvFeatureTooSmall_t, TensorNotContiguous_t,
};

const static char* exc_type[] = {
	"IndexOutOfRange", "DimNotMatch", "SizeNotEqual", "DimNotExist", "OperatorCanNotBroadcast",
	"ConvFeatureSmallerThanKernel", "TensorNotContiguous",
};

class IndexOutOfRange: public std::exception {
public:
	const static int type = ExcType::IndexOutOfRange_t;
	IndexOutOfRange(index_t idx, index_t low, index_t high, index_t dim) {
		std::printf("%s: range(%d~%d), index(%d), at dim(%d)\n", exc_type[type], low, high, idx, dim);}
};

class DimNotMatch:public std::exception {
public:
	const static int type = ExcType::DimNotMatch_t;
	DimNotMatch(index_t exp_dim, index_t got_dim) {
		std::printf("%s: expected dim(%d), got dim(%d)\n", exc_type[type], exp_dim, got_dim);}
};

class SizeNotEqual:public std::exception {
public:
	const static int type = ExcType::SizeNotEqual_t;
	SizeNotEqual(index_t size1, index_t size2) {
		std::printf("%s: size1(%d), size2(%d)\n", exc_type[type], size1, size2);}
};


class DimNotExist: public std::exception {
public:
	const static int type = ExcType::DimNotExist_t;
	DimNotExist(index_t idx, index_t low, index_t high) {
		std::printf("%s: range(%d~%d), index(%d)\n", exc_type[type], low, high, idx);}
};

class OperatorNotBroadcast: public std::exception {
public:
	const static int type = ExcType::OperatorNotBroadcast_t;
	OperatorNotBroadcast(index_t size1, index_t size2, index_t dim) {
		std::printf("%s: got size1(%d), size2(%d), at dim(%d)\n", exc_type[type], size1, size2, dim);}
};

class ConvFeatureTooSmall: public std::exception {
public:
	const static int type = ExcType::ConvFeatureTooSmall_t;
	ConvFeatureTooSmall(const std::pair<index_t, index_t>& fsize, const std::pair<index_t, index_t>& ksize) {
		std::printf("%s: feature size(%d, %d) is smaller than kernel size(%d, %d)", 
			        exc_type[type], fsize.first, fsize.second, ksize.first, ksize.second);}
};

class TensorNotContiguous: public std::exception {
public:
	const static int type = ExcType::TensorNotContiguous_t;
	TensorNotContiguous(const char* operation) {
		std::printf("%s: %s need contiguous tensor", exc_type[type], operation);}
};

} // namespace exc

#define CHECK_INDEX_IN_RANGE(idx, low, high, dim) do {	\
	if((idx) < (low) || (idx) >= (high)) throw exc::IndexOutOfRange(idx, low, high, dim);	\
} while(0)

#define CHECK_DIM_MATCH(got_dim, exp_dim) do {	\
	if((exp_dim) != (got_dim)) throw exc::DimNotMatch(exp_dim, got_dim);	\
} while(0)

#define CHECK_SIZE_EQUAL(size1, size2) do {	\
	if((size1) != (size2)) throw exc::SizeNotEqual(size1, size2);	\
} while(0)

#define CHECK_DIM_EXIST(idx, low, high) do {	\
	if((idx) < (low) || (idx) >= (high)) throw exc::DimNotExist(idx, low, high);	\
} while(0)

#define CHECK_OPERATOR_BROADCAST(roperand, loperand) do {	\
	CHECK_DIM_MATCH((roperand).dim(), (loperand).dim());	\
	for(index_t i = 0; i < (roperand).dim(); i++) 	\
		if((roperand).size(i) != (loperand).size(i) && (roperand).size(i) != 1 && (loperand).size(i) != 1)	\
			throw exc::OperatorNotBroadcast((roperand).size(i), (loperand).size(i), i);	\
} while(0)

#define CHECK_CONV_FSIZE_KSIZE(fsize, ksize) do {	\
	if((fsize).first < (ksize).first || (fsize.second) < (ksize.second))	\
		throw exc::ConvFeatureTooSmall(fsize, ksize);	\
} while(0)

#define CHECK_TENSOR_CONTIGUOUS(tensor, operation) do {	\
	if(!(tensor).is_contiguous()) throw exc::TensorNotContiguous(operation);	\
} while(0)

} // namespace el

#endif // BASE_EXCEPTION_H_
