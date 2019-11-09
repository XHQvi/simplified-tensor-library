#ifndef EXPRESSION_EXPRESSION_H_
#define EXPRESSION_EXPRESSION_H_

#include <cmath>
#include <memory>
#include <initializer_list>
#include "../utils/base.h"
#include "unary_exp.h"
#include "binary_exp.h"

namespace el {

template<typename Dtype> class Tensor;

// TODO: declare backward as private
template<typename Dtype>
class Exp {
public:
	virtual Dtype eval(index_t* ids) const = 0;
	virtual index_t dim(void) const = 0;
	virtual index_t size(index_t idx) const = 0;
	virtual void backward(void) const = 0;
	virtual ~Exp() {};
};

template<typename Dtype>
class UnaryExp: public Exp<Dtype> {
public:
	UnaryExp(const Exp<Dtype>& operand): operand_(&operand){}
	UnaryExp(const std::shared_ptr<const Exp<Dtype>>& operand): operand_(operand) {}
	virtual index_t dim(void) const {return this->operand_->dim();}
	virtual index_t size(index_t idx) const {return this->operand_->size(idx);}
protected:
	const std::shared_ptr<const Exp<Dtype>> operand_;
};

template<typename Dtype>
struct BinaryExp: public Exp<Dtype> {
public:
	BinaryExp(const Exp<Dtype>& loperand, const Exp<Dtype>& roperand): roperand_(&roperand), loperand_(&loperand){}
	BinaryExp(const std::shared_ptr<Exp<Dtype>>& loperand, const std::shared_ptr<Exp<Dtype>>& roperand): roperand_(roperand), loperand_(loperand){}
	virtual index_t dim(void) const {return this->roperand_->dim();}
	virtual index_t size(index_t idx) const {return std::max(this->roperand_->size(idx), this->loperand_->size(idx));}
protected:	
	const std::shared_ptr<const Exp<Dtype>> loperand_;
	const std::shared_ptr<const Exp<Dtype>> roperand_;
};

} // namespace el


#endif
