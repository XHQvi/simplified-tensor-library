#ifndef UTILS_BASE_H_
#define UTILS_BASE_H_

#include "exception.h"
#include "fixed_size_array.h"

namespace el {

using index_t = int;
#define INDEX_MAX INT_MAX
using IndexArray = FixedSizeArray<index_t>;

using float_t = double;
using int_t = int;
#define TENSOR_DEFAULT_TYPE float_t

} // namespace el

#endif