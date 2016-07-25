/*Copyright 2016 WaizungTaam.  All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <vector>
#include <iostream>
#include <iomanip>
#include <random>
#include <algorithm>
#include <limits>
#include <cmath>
#include <omp.h>
#include "vector.h"

Vector::Vector(int size) {
  vec = std::vector<double>(size, 0);
}
Vector::Vector(int size, double value) {
	vec = std::vector<double>(size, value);
}
Vector::Vector(int size, const std::string & mode, 
               double param_1, double param_2) {
  bool is_uniform = (mode == "uniform") || (mode == "Uniform") || 
                    (mode == "UNIFORM") || (mode == "u") || (mode == "U");
  bool is_normal = (mode == "normal") || (mode == "Normal") || 
                   (mode == "NORMAL") || (mode == "n") || (mode == "N");
  bool is_binomial = (mode == "binomial") || (mode == "Binomial") ||
                     (mode == "BINOMIAL") || (mode == "b") || (mode == "B");
  if (!(is_uniform || is_normal || is_binomial)) {
    throw "Unsupported mode";
  }
  vec = std::vector<double>(size, 0); 
  std::random_device rd;
  std::mt19937 gen(rd());
  if (is_uniform) {
    std::uniform_real_distribution<> uni_dis(param_1, param_2);
    for (double & element : vec) {
      element = uni_dis(gen);
    }
  } else if (is_normal) {
    std::normal_distribution<> nor_dis(param_1, param_2);
    for (double & element : vec) {
      element = nor_dis(gen);
    }
  } else if (is_binomial) {
    std::binomial_distribution<int> bin_dis(param_1, param_2);
    for (double & element : vec) {
      element = static_cast<double>(bin_dis(gen));
    }
  }
}
Vector::Vector(const std::vector<double> & vec_init) {
  vec = vec_init;
}
Vector::Vector(const std::initializer_list<double> & ls) : vec(ls) {
}
Vector & Vector::operator=(double value) {
	for (double & element : vec) {
		element = value;
	}
	return *this;
}
Vector & Vector::operator=(const std::vector<double> & vec_copy) {
  vec = vec_copy;
  return *this;
}
Vector & Vector::operator=(const std::initializer_list<double> & ls) {
  vec = ls;
  return *this;
}
std::vector<std::size_t> Vector::shape() const {
	std::vector<std::size_t> shape_vec(1);
	shape_vec[0] = vec.size();
	return shape_vec;
}
Vector Vector::insert(double value, int index) const {
	if (index > vec.size()) {
		throw "Out-of-range";
	}	
	Vector vec_inserted = *this;
	vec_inserted.vec.insert(vec_inserted.vec.begin() + index, value);
	return vec_inserted;
}
Vector Vector::insert(const Vector & vec_to_insert, int index) const {
	if (index > vec.size()) {
		throw "Out-of-range";
	}
	Vector vec_inserted = *this;
	for (const double element : vec_to_insert.vec) {
		vec_inserted.vec.insert(vec_inserted.vec.begin() + index, element);
		++index;
	}
	return vec_inserted;
}
Vector Vector::remove(int index) const {
	if (index >= vec.size()) {
		throw "Out-of-range";
	}
	Vector vec_removed = *this;
	vec_removed.vec.erase(vec_removed.vec.begin() + index);
	return vec_removed;
} 
Vector Vector::remove(int idx_begin, int idx_end) const {
	if (idx_begin > idx_end) {
		int tmp_swap = idx_begin;
		idx_begin = idx_end;
		idx_end = tmp_swap;
	}
	if (idx_end >= vec.size()) {
		throw "Out-of-range";
	}
	Vector vec_removed = *this;
	vec_removed.vec.erase(vec_removed.vec.begin() + idx_begin, 
                        vec_removed.vec.begin() + idx_end);
	return vec_removed;
}
Vector Vector::replace(const Vector & vec_to_replace, int index) const {
  if (index >= vec.size()) {
    return *this;
  }
  Vector vec_replaced = *this;
  int idx_rep;
  for (idx_rep = 0; idx_rep < vec_to_replace.vec.size() &&
       idx_rep + index < vec_replaced.vec.size(); ++idx_rep) {
    vec_replaced[index + idx_rep] = vec_to_replace.vec[idx_rep];
  }
  return vec_replaced;
}
Vector Vector::shuffle() const {
  std::random_device rd;
  std::mt19937 gen(rd());
  Vector vec_shuffled = *this;
  std::shuffle(vec_shuffled.vec.begin(), vec_shuffled.vec.end(), gen);
  return vec_shuffled;
}
void Vector::clear() {
  vec.clear();
}
bool Vector::empty() {
  if (vec.size() == 0) {
    return true;
  } else {
    return false;
  }
}
double Vector::sum() const {
  if (shape()[0] == 0) {
    return 0;
  }
	double sum = 0;
  // Experiments show that this method is faster then 
  // either using omp or std::accumulate
	for (const double element : vec) {
		sum += element;  
	}
	return sum;
}
double Vector::max() const {
  if (shape()[0] == 0) {
    return 0;
  }  
  double max = vec[0];
  int idx;
  // Experiments show that this is faster then std::max_element
  for (idx = 1; idx < shape()[0]; ++idx) {
    if (vec[idx] > max) {
      max = vec[idx];
    }
  }
  return max;
}
double Vector::min() const {
  if (shape()[0] == 0) {
    return 0;
  }  
  double min = vec[0];
  int idx;
  for (idx = 1; idx < shape()[0]; ++idx) {
    if (vec[idx] < min) {
      min = vec[idx];
    }
  }
  return min;
}
bool Vector::approx(const Vector & vec_to_compare, double error) const {
  int idx;
  error = std::abs(error);
  for (idx = 0; idx < vec.size(); ++idx) {
    // std::numeric_limits<double>::epsilon() returns the machine epsilon,
    // which is the difference between 1.0 and the next double value.
    // It is used here to avoid the rounding error.
    if (std::abs(vec[idx] - vec_to_compare.vec[idx]) - error > 
          std::numeric_limits<double>::epsilon()) {
      return false;
    }
  } 
  return true;
}
Vector Vector::approx(double value, double error) const {
  error = std::abs(error);
  int idx;
  Vector vec_is_approx(vec.size(), 0);
  for (idx = 0; idx < vec.size(); ++idx) {
    if (std::abs(vec[idx] - value) - error <= 
          std::numeric_limits<double>::epsilon()) {
      vec_is_approx[idx] = 1;
    }    
  }
  return vec_is_approx;
}
Vector operator+(const Vector & vec_lhs, const Vector & vec_rhs) {
	if (vec_lhs.vec.size() != vec_rhs.vec.size()) {
		throw "Inconsistent shape";
	}
	Vector vec_sum = vec_lhs;
  // size_omp is the vector size that is used to 
  // decide whether to use openmp or not.
  // Note that this is a approximate quantity obtained 
  // from experiments.
  // This is applied to all the size_omp in this file.
	const int size_omp = 300;
  int idx; 
  if (vec_sum.vec.size() < size_omp) {
    for (idx = 0; idx < vec_sum.vec.size(); ++idx) {
      vec_sum.vec[idx] = vec_sum.vec[idx] + vec_rhs.vec[idx];
    }
  } else {
    #pragma omp parallel shared(vec_sum, vec_rhs) private(idx)
    {
      #pragma omp for schedule(auto)
      for (idx = 0; idx < vec_sum.vec.size(); ++idx) {
        vec_sum.vec[idx] = vec_sum.vec[idx] + vec_rhs.vec[idx];
      }
    }
  }
  return vec_sum;
}
Vector operator+(const Vector & vec_lhs, double value) {
	Vector vec_sum = vec_lhs;
  const int size_omp = 350;
  if (vec_sum.vec.size() < size_omp) {
    for (double & element : vec_sum.vec) {
      element = element + value;
    }
  } else {
    int idx;
    #pragma omp parallel shared(vec_sum, value) private(idx)
    {
      #pragma omp for schedule(auto)
      for (idx = 0; idx < vec_sum.vec.size(); ++idx) {
        vec_sum[idx] = vec_sum[idx] + value;
      }
    }
  }
	return vec_sum;
}
Vector operator+(double value, const Vector & vec_rhs) {
  return operator+(vec_rhs, value);
}
Vector operator-(const Vector & vec_lhs, const Vector & vec_rhs) {
  return vec_lhs + ((-1) * vec_rhs);
}
Vector operator-(const Vector & vec_lhs, double value) {
  return vec_lhs + ((-1) * value);
}
Vector operator-(double value, const Vector & vec_rhs) {
  Vector vec_diff = operator-(vec_rhs, value);
  vec_diff = -1.0 * vec_diff;
  return vec_diff;
}
Vector operator*(const Vector & vec_lhs, const Vector & vec_rhs) {
  if (vec_lhs.vec.size() != vec_rhs.vec.size()) {
    throw "Inconsistent shape";
  } 
  Vector vec_prod = vec_lhs;
  const int size_omp = 300;
  int idx;
  if (vec_prod.vec.size() < size_omp) {
    for (idx = 0; idx < vec_prod.vec.size(); ++idx) {
      vec_prod.vec[idx] = vec_prod.vec[idx] * vec_rhs.vec[idx];
    }
  } else {
    #pragma omp parallel shared(vec_prod, vec_rhs) private(idx)
    {
      #pragma omp for schedule(auto)
      for (idx = 0; idx < vec_prod.vec.size(); ++idx) {
        vec_prod.vec[idx] = vec_prod.vec[idx] * vec_rhs.vec[idx];
      }      
    }
  }
  return vec_prod;
}
Vector operator*(const Vector & vec_lhs, double value) {
  Vector vec_prod = vec_lhs;
  const int size_omp = 350;
  if (vec_prod.vec.size() < size_omp) {
    for (double & element : vec_prod.vec) {
      element = element * value;
    }
  } else {
    int idx;
    #pragma omp parallel shared(vec_prod, value) private(idx)
    {
      #pragma omp for schedule(auto)
      for (idx = 0; idx < vec_prod.vec.size(); ++idx) {
        vec_prod[idx] = vec_prod[idx] * value;
      }
    }
  }
  return vec_prod;  
}
Vector operator*(double value, const Vector & vec_rhs) {
  return operator*(vec_rhs, value);
}
Vector operator/(const Vector & vec_lhs, const Vector & vec_rhs) {
  if (vec_lhs.vec.size() != vec_rhs.vec.size()) {
    throw "Inconsistent shape";
  } 
  Vector vec_quot = vec_lhs;   
  const int size_omp = 300;
  int idx;
  if (vec_quot.vec.size() < size_omp) {
    for (idx = 0; idx < vec_quot.vec.size(); ++idx) {
      vec_quot.vec[idx] = static_cast<double>(vec_quot.vec[idx]) 
                          / vec_rhs.vec[idx];
    }
  } else {
    #pragma omp parallel shared(vec_quot, vec_rhs) private(idx)
    {
      #pragma omp for schedule(auto)
      for (idx = 0; idx < vec_quot.vec.size(); ++idx) {
        vec_quot.vec[idx] = static_cast<double>(vec_quot.vec[idx]) 
                            / vec_rhs.vec[idx];
      }      
    }
  }
  return vec_quot;
}
Vector operator/(const Vector & vec_lhs, double value) {
  return vec_lhs * (1.0 / static_cast<double>(value));
}
Vector operator/(double value, const Vector & vec_rhs) {
  Vector vec_quot = vec_rhs; 
  const int size_omp = 350;
  if (vec_quot.vec.size() < size_omp) {
    for (double & element : vec_quot.vec) {
      element = static_cast<double>(value) / element;
    }
  } else {
    int idx;
    #pragma omp parallel shared(vec_quot, value) private(idx)
    {
      #pragma omp for schedule(auto)
      for (idx = 0; idx < vec_quot.vec.size(); ++idx) {
        vec_quot[idx] = static_cast<double>(value) / vec_quot[idx];
      }
    }
  }
  return vec_quot;
}
void Vector::operator+=(double value) {
	(*this) = (*this) + value;
}
void Vector::operator+=(const Vector & vec_to_add) {
  (*this) = (*this) + vec_to_add;
}
void Vector::operator-=(double value) {
	(*this) = (*this) - value;
}
void Vector::operator-=(const Vector & vec_to_sub) {
  (*this) = (*this) - vec_to_sub;
}
void Vector::operator*=(double value) {
	(*this) = (*this) * value;
}
void Vector::operator*=(const Vector & vec_to_mul) {
  (*this) = (*this) * vec_to_mul;
}
void Vector::operator/=(double value) {
	(*this) = (*this) / value;
}
void Vector::operator/=(const Vector & vec_to_div) {
  (*this) = (*this) / vec_to_div;
}
bool operator==(const Vector & vec_lhs, const Vector & vec_rhs) {
	if (vec_lhs.vec.size() != vec_rhs.vec.size()) {
		return false;
	}
	int idx;
	for (idx = 0; idx < vec_lhs.vec.size(); ++idx) {
		if (!::approx(vec_lhs.vec[idx], vec_rhs.vec[idx])) {
			return false;
		}
	}
	return true;
}
Vector operator==(const Vector & vec_lhs, double value) {
  Vector vec_is_equ(vec_lhs.shape()[0], 0);   
  const int size_omp = 360;  
  int idx;
  if (vec_lhs.vec.size() < size_omp) {
    for (idx = 0; idx < vec_is_equ.shape()[0]; ++idx) {
      if (::approx(vec_lhs[idx], value)) {
        vec_is_equ[idx] = 1.0;
      }
    }    
  } else {
    #pragma omp parallel shared(vec_is_equ, value) private(idx)
    {
      #pragma omp for schedule(auto)
      for (idx = 0; idx < vec_is_equ.shape()[0]; ++idx) {
        if (::approx(vec_lhs[idx], value)) {
          vec_is_equ[idx] = 1.0;
        }
      }  
    }
  }
  return vec_is_equ;
}
Vector operator==(double value, const Vector & vec_rhs) {
  return vec_rhs == value;
}
bool operator!=(const Vector & vec_lhs, const Vector & vec_rhs) {
  return !(vec_lhs == vec_rhs);
}
Vector operator!=(const Vector & vec_lhs, double value) {
  Vector vec_is_equ(vec_lhs.shape()[0], 0);
  const int size_omp = 360;  
  int idx;
  if (vec_lhs.vec.size() < size_omp) {
    for (idx = 0; idx < vec_is_equ.shape()[0]; ++idx) {
      if (!::approx(value, vec_lhs[idx])) {
        vec_is_equ[idx] = 1.0;
      }
    }    
  } else {
    #pragma omp parallel shared(vec_is_equ, value) private(idx)
    {
      #pragma omp for schedule(auto)
      for (idx = 0; idx < vec_is_equ.shape()[0]; ++idx) {
        if (!::approx(value, vec_lhs[idx])) {
          vec_is_equ[idx] = 1.0;
        }
      }  
    }
  }
  return vec_is_equ;
}
Vector operator!=(double value, const Vector & vec_rhs) {
  return vec_rhs != value;
}
Vector operator<(const Vector & vec_lhs, const Vector & vec_rhs) {
  if (vec_lhs.shape()[0] != vec_rhs.shape()[0]) {
    throw "Inconsistent shape";
  }
  Vector vec_is_les(vec_lhs.shape()[0]);
  const int size_omp = 400;
  int idx;
  if (vec_is_les.vec.size() < size_omp) {
    for (idx = 0; idx < vec_is_les.shape()[0]; ++idx) {
      if (vec_lhs.vec[idx] < vec_rhs.vec[idx]) {
        vec_is_les.vec[idx] = 1.0;
      }
    }    
  } else {
    #pragma omp parallel shared(vec_lhs, vec_rhs, vec_is_les) private(idx)
    {
      #pragma omp for schedule(auto)
      for (idx = 0; idx < vec_is_les.shape()[0]; ++idx) {
        if (vec_lhs.vec[idx] < vec_rhs.vec[idx]) {
          vec_is_les.vec[idx] = 1.0;
        }
      }       
    }
  }
  return vec_is_les;
}
Vector operator<(const Vector & vec_lhs, double value) {
  Vector vec_is_les(vec_lhs.shape()[0]);
  const int size_omp = 400;
  int idx;
  if (vec_is_les.vec.size() < size_omp) {
    for (idx = 0; idx < vec_is_les.shape()[0]; ++idx) {
      if (vec_lhs.vec[idx] < value) {
        vec_is_les.vec[idx] = 1.0;
      }
    }    
  } else {
    #pragma omp parallel shared(vec_lhs, value, vec_is_les) private(idx)
    {
      #pragma omp for schedule(auto)
      for (idx = 0; idx < vec_is_les.shape()[0]; ++idx) {
        if (vec_lhs.vec[idx] < value) {
          vec_is_les.vec[idx] = 1.0;
        }
      }       
    }
  }
  return vec_is_les;
}
Vector operator<(double value, const Vector & vec_rhs) {
  Vector vec_is_les(vec_rhs.shape()[0]);
  const int size_omp = 400;
  int idx;
  if (vec_is_les.vec.size() < size_omp) {
    for (idx = 0; idx < vec_is_les.shape()[0]; ++idx) {
      if (value < vec_rhs.vec[idx]) {
        vec_is_les.vec[idx] = 1.0;
      }
    }    
  } else {
    #pragma omp parallel shared(vec_rhs, value, vec_is_les) private(idx)
    {
      #pragma omp for schedule(auto)
      for (idx = 0; idx < vec_is_les.shape()[0]; ++idx) {
        if (value < vec_rhs.vec[idx]) {
          vec_is_les.vec[idx] = 1.0;
        }
      }       
    }
  }
  return vec_is_les;
}
Vector operator<=(const Vector & vec_lhs, const Vector & vec_rhs) {
  if (vec_lhs.shape()[0] != vec_rhs.shape()[0]) {
    throw "Inconsistent shape";
  }
  Vector vec_is_leq(vec_lhs.shape()[0]);
  const int size_omp = 400;
  int idx;
  if (vec_is_leq.vec.size() < size_omp) {
    for (idx = 0; idx < vec_is_leq.shape()[0]; ++idx) {
      if (vec_lhs.vec[idx] <= vec_rhs.vec[idx]) {
        vec_is_leq.vec[idx] = 1.0;
      }
    }    
  } else {
    #pragma omp parallel shared(vec_lhs, vec_rhs, vec_is_leq) private(idx)
    {
      #pragma omp for schedule(auto)
      for (idx = 0; idx < vec_is_leq.shape()[0]; ++idx) {
        if (vec_lhs.vec[idx] <= vec_rhs.vec[idx]) {
          vec_is_leq.vec[idx] = 1.0;
        }
      }       
    }
  }
  return vec_is_leq;
}
Vector operator<=(const Vector & vec_lhs, double value) {
  Vector vec_is_leq(vec_lhs.shape()[0]);
  const int size_omp = 400;
  int idx;
  if (vec_is_leq.vec.size() < size_omp) {
    for (idx = 0; idx < vec_is_leq.shape()[0]; ++idx) {
      if (vec_lhs.vec[idx] <= value) {
        vec_is_leq.vec[idx] = 1.0;
      }
    }    
  } else {
    #pragma omp parallel shared(vec_lhs, value, vec_is_leq) private(idx)
    {
      #pragma omp for schedule(auto)
      for (idx = 0; idx < vec_is_leq.shape()[0]; ++idx) {
        if (vec_lhs.vec[idx] <= value) {
          vec_is_leq.vec[idx] = 1.0;
        }
      }       
    }
  }
  return vec_is_leq;
}
Vector operator<=(double value, const Vector & vec_rhs) {
  Vector vec_is_leq(vec_rhs.shape()[0]);
  const int size_omp = 400;
  int idx;
  if (vec_is_leq.vec.size() < size_omp) {
    for (idx = 0; idx < vec_is_leq.shape()[0]; ++idx) {
      if (value <= vec_rhs.vec[idx]) {
        vec_is_leq.vec[idx] = 1.0;
      }
    }    
  } else {
    #pragma omp parallel shared(vec_rhs, value, vec_is_leq) private(idx)
    {
      #pragma omp for schedule(auto)
      for (idx = 0; idx < vec_is_leq.shape()[0]; ++idx) {
        if (value <= vec_rhs.vec[idx]) {
          vec_is_leq.vec[idx] = 1.0;
        }
      }       
    }
  }
  return vec_is_leq;
}
Vector operator>(const Vector & vec_lhs, const Vector & vec_rhs) {
  return vec_rhs < vec_lhs;
}
Vector operator>(const Vector & vec_lhs, double value) {
  return value < vec_lhs;
}
Vector operator>(double value, const Vector & vec_rhs) {
  return vec_rhs < value;
}
Vector operator>=(const Vector & vec_lhs, const Vector & vec_rhs) {
  return vec_rhs <= vec_lhs;
}
Vector operator>=(const Vector & vec_lhs, double value) {
  return value <= vec_lhs;
}
Vector operator>=(double value, const Vector & vec_rhs) {
  return vec_rhs <= value;
}
double & Vector::operator[](int index) {
  if (index >= 0) {
    if (index >= vec.size()) {
      throw "Out-of-range";
    }
    return vec.at(index);
  } else if (index < 0) {
    if ((-1) * index > vec.size()) {
      throw "Out-of-range";
    }
    return vec.at(vec.size() + index);
  }
}
const double & Vector::operator[](int index) const {
  if (index >= 0) {
    if (index >= vec.size()) {
      throw "Out-of-range";
    }
    return vec.at(index);
  } else if (index < 0) {
    if ((-1) * index > vec.size()) {
      throw "Out-of-range";
    }
    return vec.at(vec.size() + index);
  }
}
Vector Vector::operator()(int idx_begin, int idx_end) const {
  if (idx_begin > idx_end) {
  	int tmp_swap = idx_begin;
  	idx_begin = idx_end;
  	idx_end = tmp_swap;
  }
  if (idx_end > vec.size()) {
  	throw "Out-of-range";
  }
  Vector vec_partial(idx_end - idx_begin);
  vec_partial.vec = std::vector<double>(vec.begin() + idx_begin, 
                                        vec.begin() + idx_end);
  return vec_partial;
}
std::ostream & operator<<(std::ostream & os, const Vector & vec_os) {
  os << "[";
  int idx = 0;
  for (const double element : vec_os.vec) {
    os << std::setw(15) << std::setprecision(8) << std::setfill(' ')
       << std::scientific << std::left << std::showpos
       << element;
    if (idx != vec_os.vec.size() - 1) {
      os << " ";
    }
    ++idx;
  }
  os << std::resetiosflags(std::ios_base::scientific) 
     << std::resetiosflags(std::ios_base::right)
     << std::resetiosflags(std::ios_base::showpos);
  os << "]";
  return os;
}
std::istream & operator>>(std::istream & is, Vector & vec_is) {
	if (vec_is.vec.size() == 0) {
		double element;
		while (is >> element) {
			vec_is.vec.push_back(element);
		}
	} else {
		for (double & element : vec_is.vec) {
			is >> element;
		}
	}
	return is;
}
