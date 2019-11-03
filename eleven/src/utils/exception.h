#ifndef UTILS_EXCEPTION_H_
#define UTILS_EXCEPTION_H_

#include <iostream>
#include <sstream>

namespace el {

#define ERROR_LOCATION __FILE__, __func__, __LINE__
#define THROW_ERROR(err_cls, msg)   \
    throw err::err_cls(msg, ERROR_LOCATION)

#define CHECK_EQUAL(x, y, err_cls, msg) \
    if((x) != (y)) THROW_ERROR(err_cls, msg)
#define CHECK_BETWEEN(idx, low, high, err_cls, msg) \
    if((idx) < (low) || (idx) >= (high)) THROW_ERROR(err_cls, msg)
#define CHECK_TRUE(cond, err_cls, msg)  \
    if(!(cond)) THROW_ERROR(err_cls, msg)



namespace err {

class Error: public std::exception {
public:
    Error(const char* type, const char* msg, const char* file, const char* func, unsigned int line) 
        : file_(file), func_(func), line_(line), type_(type), msg_(msg) {};
    const char* what() const noexcept {
        std::ostringstream out;
        out << file_ << ", function " << func_ << ", line " << line_ << ":" << std::endl;
        out << type_ << ": " << msg_ << std::endl;
        return out.str().c_str();
    }
private:
    std::string type_;
    std::string msg_;
    std::string file_;
    std::string func_;
    unsigned int line_;
};

class IndexOutOfRange: public Error {
public:
    IndexOutOfRange(const char* msg, const char* file, const char* func, unsigned int line)
        :Error("IndexOutOfRange", msg, file, func, line) {}
};


}  // namespace err
}  // namespace el


#endif
