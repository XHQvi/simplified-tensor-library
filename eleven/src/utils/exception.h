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
public:
    IndexOutOfRange(const char* file, const char* func, unsigned int line);
};

class DimNotMatch: public Error {
public:
	DimNotMatch(const char* file, const char* func, unsigned int line);
};

class OpCondNotMet: public Error {
public:
	OpCondNotMet(const char* file, const char* func, unsigned int line);
};

}  // namespace err

#define ERROR_LOCATION __FILE__, __func__, __LINE__
#define THROW_ERROR(err_cls, format, ...)	do {	\
    std::sprintf(err::msg, format, ##__VA_ARGS__);    \
    throw err::err_cls(ERROR_LOCATION);	\
} while(0)


#define CHECK_EQUAL(x, y, err_cls, format, ...) \
    if((x) != (y)) THROW_ERROR(err_cls, format, ##__VA_ARGS__)

#define CHECK_BETWEEN(idx, low, high, err_cls, format, ...) \
    if((idx) < (low) || (idx) >= (high)) THROW_ERROR(err_cls, format, ##__VA_ARGS__)

#define CHECK_TRUE(cond, err_cls, format, ...)  \
    if(!(cond)) THROW_ERROR(err_cls, format, ##__VA_ARGS__)

}  // namespace el


#endif
