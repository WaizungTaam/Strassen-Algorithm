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

#ifndef VECTOR_H
#define VECTOR_H

#include <vector>
#include <iostream>
#include <limits>
#include <cmath>
#include <string>

class Vector {
public:
  Vector() = default;
  Vector(const Vector &) = default;
  Vector(Vector &&) = default;
  Vector & operator=(const Vector &) = default;
  Vector & operator=(Vector &&) = default;
  ~Vector() = default;
  Vector(int);
  Vector(int, double);
  Vector(int, const std::string &, double, double);
  Vector(const std::vector<double> &);
  Vector(const std::initializer_list<double> &);
  Vector & operator=(double);
  Vector & operator=(const std::vector<double> &);
  Vector & operator=(const std::initializer_list<double> &);
  std::vector<std::size_t> shape() const;
  Vector insert(double, int) const;
  Vector insert(const Vector &, int) const;
  Vector remove(int) const;
  Vector remove(int, int) const;
  Vector replace(const Vector &, int) const;
  Vector shuffle() const;
  void clear();
  bool empty();
  double sum() const;
  double max() const;
  double min() const;
  bool approx(const Vector &, double) const;
  Vector approx(double, double) const;
  friend Vector operator+(const Vector &, const Vector &);
  friend Vector operator+(const Vector &, double);
  friend Vector operator+(double, const Vector &);
  friend Vector operator-(const Vector &, const Vector &);
  friend Vector operator-(const Vector &, double);
  friend Vector operator-(double, const Vector &);
  friend Vector operator*(const Vector &, const Vector &);
  friend Vector operator*(const Vector &, double);
  friend Vector operator*(double, const Vector &);
  friend Vector operator/(const Vector &, const Vector &);
  friend Vector operator/(const Vector &, double);
  friend Vector operator/(double, const Vector &);
  void operator+=(double);
  void operator+=(const Vector &);
  void operator-=(double);
  void operator-=(const Vector &);
  void operator*=(double);
  void operator*=(const Vector &);
  void operator/=(double);
  void operator/=(const Vector &);
  friend bool operator==(const Vector &, const Vector &);
  friend Vector operator==(const Vector &, double);
  friend Vector operator==(double, const Vector &);
  friend bool operator!=(const Vector &, const Vector &);
  friend Vector operator!=(const Vector &, double);
  friend Vector operator!=(double, const Vector &);
  friend Vector operator<(const Vector &, const Vector &);
  friend Vector operator<(const Vector &, double);
  friend Vector operator<(double, const Vector &);
  friend Vector operator<=(const Vector &, const Vector &);
  friend Vector operator<=(const Vector &, double);
  friend Vector operator<=(double, const Vector &);
  friend Vector operator>(const Vector &, const Vector &);
  friend Vector operator>(const Vector &, double);
  friend Vector operator>(double, const Vector &);
  friend Vector operator>=(const Vector &, const Vector &);
  friend Vector operator>=(const Vector &, double);
  friend Vector operator>=(double, const Vector &);  
  double & operator[](int);
  const double & operator[](int) const;
  Vector operator()(int, int) const;
  friend std::ostream & operator<<(std::ostream & out, const Vector &);
  friend std::istream & operator>>(std::istream & in, Vector &);
private:
  std::vector<double> vec;
};

template<class T>
typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type
approx(T x, T y) {
  // std::numeric_limits<double>::epsilon() returns the machine epsilon,
  // here it is scaled to the magnitude of (x + y) and used to judge 
  // whether x is equal to y while avoiding the rounding error.
  return std::abs(x - y) < std::numeric_limits<T>::epsilon() * std::abs(x + y) ||
         std::abs(x - y) < std::numeric_limits<T>::min();
}

#endif // vector.h
