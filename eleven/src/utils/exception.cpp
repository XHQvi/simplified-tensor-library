#include "exception.h"

namespace el {
namespace err {

char msg[300];

Error::Error(const char* type, const char* file, const char* func, unsigned int line) 
    : file_(file), func_(func), line_(line), type_(type) {};
const char* Error::what() const noexcept {
    std::ostringstream out;
    out << file_ << ", in function " << func_ << "(), line " << line_ << ":" << std::endl;
    out << type_ << ": " << msg << std::endl;
    return out.str().c_str();
}

IndexOutOfRange::IndexOutOfRange(const char* file, const char* func, unsigned int line)
    : Error("IndexOutOfRange", file, func, line) {}

DimNotMatch::DimNotMatch(const char* file, const char* func, unsigned int line)
	: Error("DimNotMatch", file, func, line) {}

TensorNotContiguous::TensorNotContiguous(const char* file, const char* func, unsigned int line)
	: Error("TensorNotContiguous", file, func, line) {}

DsizeNotMatch::DsizeNotMatch(const char* file, const char* func, unsigned int line)
	: Error("DsizeNotMatch", file, func, line) {}

OperandSizeNotMatch::OperandSizeNotMatch(const char* file, const char* func, unsigned int line)
	: Error("OperandSizeNotMatch", file, func, line) {}

}  // namespace err
}  // namespace el