#ifndef BASE_EXCEPTION_H_
#define BASE_EXCEPTION_H_

#include <exception>
#include <cstdio>
#include "type.h"

namespace el {
namespace exc {

enum ExcType : int{
	IndexOutOfRange_t = 0, DimNotMatch_t, DimNotExist_t,
};

const static char* exc_type[] = {
	"IndexOutOfRange", "DimNotMatch", "DimNotExist", 
};

class IndexOutOfRange: public std::exception {
public:
	const static int type = ExcType::IndexOutOfRange_t;
	IndexOutOfRange(index_t idx, index_t low, index_t high, index_t dim) {
		std::printf("%s: range(%d~%d), index(%d), dim(%d)\n", exc_type[type], low, high, idx, dim);}
};

class DimNotMatch:public std::exception {
public:
	const static int type = ExcType::DimNotMatch_t;
	DimNotMatch(index_t exp_dim, index_t got_dim) {
		std::printf("%s: expected dim(%d), got dim(%d)\n", exc_type[type], exp_dim, got_dim);}
};

class DimNotExist: public std::exception {
public:
	const static int type = ExcType::DimNotExist_t;
	DimNotExist(index_t idx, index_t low, index_t high) {
		std::printf("%s: range(%d~%d), index(%d)\n", exc_type[type], low, high, idx);}
};
} // namespace exc

#define CHECK_INDEX_IN_RANGE(idx, low, high, dim) do {	\
	if((idx) < (low) || (idx) >= (high)) throw exc::IndexOutOfRange(idx, low, high, dim);	\
} while(0)

#define CHECK_DIM_MATCH(got_dim, exp_dim) do {	\
	if((exp_dim) != (got_dim)) throw exc::DimNotMatch(exp_dim, got_dim);	\
} while(0)

#define CHECK_DIM_EXIST(idx, low, high) do {	\
	if((idx) < (low) || (idx) >= (high)) throw exc::DimNotExist(idx, low, high);	\
} while(0)

} // namespace el

#endif // BASE_EXCEPTION_H_
