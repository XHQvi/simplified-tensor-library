#ifndef BASE_TYPE_H_
#define BASE_TYPE_H_

namespace el {

typedef int index_t;

namespace DataType {  // data type
typedef int int_t;
typedef double float_t;
} // namespace dt

#define TENSOR_DEFAULT_TYPE DataType::float_t

} // namespace el
#endif // BASE_TYPE_H_
