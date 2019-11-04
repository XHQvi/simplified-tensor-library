#ifndef UTILS_EXCEPTION_H_
#define UTILS_EXCEPTION_H_

#include <iostream>
#include <sstream>
#include <cstdio>

namespace el {
namespace err {

extern char msg[300];

class Error: public std::exception {
public:
    Error(const char* type, const char* file, const char* func, unsigned int line);
    const char* what() const noexcept;
private:
    std::string type_;
    std::string file_;
    std::string func_;
    unsigned int line_;
};

class IndexOutOfRange: public Error {
    public: IndexOutOfRange(const char* file, const char* func, unsigned int line);
};
class DimNotMatch: public Error {
    public:	DimNotMatch(const char* file, const char* func, unsigned int line);
};
class TensorNotContiguous: public Error {
    public:	TensorNotContiguous(const char* file, const char* func, unsigned int line);
};
class DsizeNotMatch: public Error {
    public: DsizeNotMatch(const char* file, const char* func, unsigned int line);
};
class OperandSizeNotMatch: public Error {
    public: OperandSizeNotMatch(const char* file, const char* func, unsigned int line);
};

}  // namespace err

#define ERROR_LOCATION __FILE__, __func__, __LINE__
#define THROW_ERROR(err_cls, format, ...)	do {	\
    std::sprintf(err::msg, format, ##__VA_ARGS__);    \
    throw err::err_cls(ERROR_LOCATION);	\
} while(0)


// base assert macro
#define CHECK_EQUAL(x, y, err_cls, format, ...) \
    if((x) != (y)) THROW_ERROR(err_cls, format, ##__VA_ARGS__)

#define CHECK_BETWEEN(idx, low, high, err_cls, format, ...) \
    if((idx) < (low) || (idx) >= (high)) THROW_ERROR(err_cls, format, ##__VA_ARGS__)

#define CHECK_TRUE(cond, err_cls, format, ...)  \
    if(!(cond)) THROW_ERROR(err_cls, format, ##__VA_ARGS__)


// higher level assert macro
#define CHECK_BROADCAST(roperand, loperand)    do {    \
    CHECK_EQUAL((roperand).dim(), (loperand).dim(), OperandSizeNotMatch,    \
        "The operands' dim should be same, but got %dD and %dD", (roperand).dim(), (loperand).dim());   \
    for(index_t ii = 0; ii < (roperand).dim(); ii++)   \
        if((roperand).size(ii) != (loperand).size(ii) && (roperand).size(ii) != 1 && (loperand).size(ii) != 1)  \
            THROW_ERROR(OperandSizeNotMatch,    \
                "Operands' size on %d dimension, %d and %d, can't be broadcasted.",     \
                ii, (roperand).size(ii), (loperand).size(ii));  \
} while(0)

}  // namespace el


#endif
