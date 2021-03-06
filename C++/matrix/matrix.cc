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
#include <string>
#include <random>
#include <algorithm>
#include <fstream>
#include <iterator>
#include <sstream>
#include <omp.h>
#include "vector.h"
#include "matrix.h"

Matrix::Matrix(int num_rows, int num_cols) {
  mat = std::vector<Vector>(num_rows, Vector(num_cols, 0));
}
Matrix::Matrix(const std::vector<std::size_t> & shape_vec) {
  if (shape_vec.size() < 2) {
    throw "Invalid shape vector for Matrix";
  }
  mat = Matrix(shape_vec[0], shape_vec[1]).mat;
}
Matrix::Matrix(int num_rows, int num_cols, double value) {
  mat = std::vector<Vector>(num_rows, Vector(num_cols, value));
}
Matrix::Matrix(const std::vector<std::size_t> & shape_vec, double value) {
  if (shape_vec.size() < 2) {
    throw "Invalid shape vector for Matrix";
  }
  mat = Matrix(shape_vec[0], shape_vec[1], value).mat;  
}
Matrix::Matrix(int num_rows, int num_cols, 
               const std::string & mode, double param_1, double param_2) {
  bool is_uniform = (mode == "uniform") || (mode == "Uniform") || 
                    (mode == "UNIFORM") || (mode == "u") || (mode == "U");
  bool is_normal = (mode == "normal") || (mode == "Normal") || 
                   (mode == "NORMAL") || (mode == "n") || (mode == "N");
  bool is_binomial = (mode == "binomial") || (mode == "Binomial") ||
                     (mode == "BINOMIAL") || (mode == "b") || (mode == "B");                   
  if (!(is_uniform || is_normal || is_binomial)) {
    throw "Unsupported mode";
  }
  std::vector<Vector> mat_init(num_rows, Vector(num_cols, 0));
  mat = mat_init;
  std::random_device rd;
  std::mt19937 gen(rd());
  if (is_uniform) {
    std::uniform_real_distribution<> uni_dis(param_1, param_2);
    int idx_row, idx_col;
    for (idx_row = 0; idx_row < num_rows; ++idx_row) {
      for (idx_col = 0; idx_col < num_cols; ++idx_col) {
        mat[idx_row][idx_col] = uni_dis(gen);
      }
    }
  } else if (is_normal) {
    std::normal_distribution<> nor_dis(param_1, param_2);
    int idx_row, idx_col;
    for (idx_row = 0; idx_row < num_rows; ++idx_row) {
      for (idx_col = 0; idx_col < num_cols; ++idx_col) {
        mat[idx_row][idx_col] = nor_dis(gen);
      }
    }
  } else if (is_binomial) {
    std::binomial_distribution<int> bin_dis(param_1, param_2);
    int idx_row, idx_col;
    for (idx_row = 0; idx_row < num_rows; ++idx_row) {
      for (idx_col = 0; idx_col < num_cols; ++idx_col) {
        mat[idx_row][idx_col] = bin_dis(gen);
      }
    }
  }
}
Matrix::Matrix(const std::vector<std::size_t> & shape_vec, 
               const std::string & mode, double param_1, double param_2) {
  if (shape_vec.size() < 2) {
    throw "Invalid shape vector for Matrix";
  }  
  mat = Matrix(shape_vec[0], shape_vec[1], mode, param_1, param_2).mat;
}
Matrix::Matrix(const Vector & v) {
  int idx;
  for (idx = 0; idx < v.shape()[0]; ++idx) {
    Vector row(1, v[idx]);
    mat.push_back(row);
  }
}
Matrix::Matrix(const std::vector<std::vector<double>> & vec_2d) {
  for (const std::vector<double> row : vec_2d) {
    mat.push_back(Vector(row));
  }
}
Matrix::Matrix(const std::initializer_list< 
               std::initializer_list<double>> & ls) {
  for (auto row_ls : ls) {
    Vector row_mat = row_ls;
    mat.push_back(row_mat);
  }
}
Matrix::Matrix(const std::string & file_name) {
  std::ifstream file(file_name);
  std::string line_str;
  if (file.is_open()) {
    while (std::getline(file, line_str)) {
      std::istringstream buffer(line_str);
      std::vector<double> line_dbl((std::istream_iterator<double>(buffer)),
                                   std::istream_iterator<double>());
      mat.push_back(Vector(line_dbl));
    }
  } else {
    throw "Cannot open " + file_name;
  }  
}
Matrix & Matrix::operator=(const Vector & v) {
  if (shape()[0] == 0) {
    std::vector<Vector> mat_init(v.shape()[0], Vector(1, 0));
    int idx_row;
    for (idx_row = 0; idx_row < mat_init.size(); ++idx_row) {
      mat_init[idx_row] = v[idx_row];
    }
    mat = mat_init;
  } else if (shape()[0] == 1 && shape()[1] == v.shape()[0]) {
    mat[0] = v;
    return *this;
  } else if ((shape()[1] == 1 && shape()[0] == v.shape()[0])) {
    int idx_row;
    for (idx_row = 0; idx_row < shape()[0]; ++idx_row) {      
      mat[idx_row] = v[idx_row];
    }
    return *this;
  } else {
    throw "Inconsistent shape";
  }
}
Matrix & Matrix::operator=(double value) {
  if (shape()[0] == 0) {
    std::vector<Vector> mat_init(1, Vector(1, value));
    mat = mat_init;
  } else {
    for (Vector & row : mat) {
      row = value;
    }
  }
  return *this;
}
Matrix & Matrix::operator=(const std::initializer_list< 
                           std::initializer_list<double>> & ls) {
  if (shape()[0] == 0) {
    for (const auto row_ls : ls) {
      Vector row_mat = row_ls;
      mat.push_back(row_mat);
    }    
  }else if (shape()[0] != ls.size()) {
    throw "Inconsistent shape";
  } else {
      int idx_row = 0;
    for (const auto row_ls : ls) {
      Vector row_mat = row_ls;
      mat[idx_row] = row_mat;
      ++idx_row;
    }
  }
  return *this;
}
std::vector<std::size_t> Matrix::shape() const {
  std::vector<std::size_t> shape(2, 0);
  shape[0] = mat.size();
  if (mat.size() == 0) {
    shape[1] = 0;
  } else {
    shape[1] = mat[0].shape()[0];
  }
  return shape;
}
Matrix Matrix::insert(const Vector & v, int dim, int index) const {
  if (dim == 0) {
    if (v.shape()[0] != shape()[1]) {
      throw "Inconsistent size";
    }
    if (index > shape()[0]) {
      throw "Out-of-range";
    }
    Matrix res = *this;
    res.mat.insert(res.mat.begin() + index, v);
    return res;
  } else if (dim == 1) {
    if (v.shape()[0] != shape()[0]) {
      throw "Inconsistent size";
    }
    if (index > shape()[1]) {
      throw "Out-of-range";
    }
    Matrix res = *this;
    int idx_row;
    for (idx_row = 0; idx_row < res.shape()[0]; ++idx_row) {
      res.mat[idx_row] = res.mat[idx_row].insert(
        v[idx_row], index);
    }
    return res;
  } else {
    throw "Invalid dimension";
  }
}
Matrix Matrix::insert(const Matrix & m, int dim, int index) const {
  if (dim == 0) {
    if (m.shape()[1] != shape()[1]) {
      throw "Inconsistent size";
    }
    if (index > shape()[0]) {
      throw "Out-of-range";
    }
    Matrix res = *this;
    for (const Vector row_m : m.mat) {
      res.mat.insert(res.mat.begin() + index, row_m);
      ++index;
    }
    return res;
  } else if (dim == 1) {
    if (m.shape()[0] != shape()[0]) {
      throw "Inconsistent size";
    }
    if (index > shape()[1]) {
      throw "Out-of-range";
    }
    Matrix res = *this;
    int idx_row, idx_col;
    for (idx_row = 0; idx_row < res.shape()[0]; ++idx_row) {
      for (idx_col = 0; idx_col < m.shape()[1]; ++idx_col) {
        res.mat[idx_row] = res.mat[idx_row].insert(
          m.mat[idx_row][idx_col], index + idx_col);
      }
    }
    return res;
  } else {
    throw "Invalid dimension";
  }
}
Matrix Matrix::remove(int dim, int index) const {
  if (dim == 0) {
    if (index >= shape()[0]) {
      throw "Out-of-range";
    }
    Matrix res = *this;
    res.mat.erase(res.mat.begin() + index);
    return res;
  } else if (dim == 1) {
    if (index >= shape()[1]) {
      throw "Out-of-range";
    }
    Matrix res = *this;
    for (Vector & row : res.mat) {
      row = row.remove(index);
    }
    return res;
  } else {
    throw "Invalid dimension";
  }
}
Matrix Matrix::remove(int dim, int idx_begin, int idx_end) const {
  if (idx_begin > idx_end) {
    int tmp_swap = idx_begin;
    idx_begin = idx_end;
    idx_end = tmp_swap;
  }
  if (dim == 0) {
    if (idx_end >= shape()[0]) {
      throw "Out-of-range";
    }
    Matrix res = *this;
    res.mat.erase(res.mat.begin() + idx_begin,
                  res.mat.begin() + idx_end);
    return res;
  } else if (dim == 1) {
    if (idx_end >= shape()[1]) {
      throw "Out-of-range";
    }
    Matrix res = *this;
    for (Vector & row : res.mat) {
      row = row.remove(idx_begin, idx_end);
    }
    return res;
  } else {
    throw "Invalid dimension";
  }
}
Matrix Matrix::replace(const Vector & v, int idx_m, int idx_v, int dim) const {
  if (dim == 0) {
    if (idx_m >= shape()[0]) {
      return *this;
    }
    Matrix res = *this;
    res.mat[idx_m] = res.mat[idx_m].replace(v, idx_v);
    return res;
  } else if (dim == 1) {
    if (idx_m >= shape()[1]) {
      return *this;
    }
    Matrix res = *this;
    int idx;
    for (idx = 0; idx < v.shape()[0] && idx_v + idx < res.shape()[0]; ++idx) {
      res.mat[idx][idx_m] = v[idx];
    }
    return res;
  } else {
    throw "Invalid dimension";
  }
}
Matrix Matrix::replace(const Matrix & m, int idx_r_begin, 
                                         int idx_c_begin) const {
  int idx_r, idx_c;
  Matrix res = *this;
  for (idx_r = 0; idx_r < m.shape()[0] && 
       idx_r_begin + idx_r < res.shape()[0]; ++idx_r) {
    for (idx_c = 0; idx_c < m.shape()[1] &&
         idx_c_begin + idx_c < res.shape()[1]; ++idx_c) {
      res.mat[idx_r_begin + idx_r][idx_c_begin + idx_c] = m.mat[idx_r][idx_c];
    }
  }
  return res;
}
Matrix Matrix::shuffle() const {
  std::random_device rd;
  std::mt19937 gen(rd());
  Matrix res = *this;
  std::shuffle(res.mat.begin(), res.mat.end(), gen);
  return res;
}
void Matrix::clear() {
  mat.clear();
}
bool Matrix::empty() const {
  if (shape()[0] == 0) {
    return true;
  } else {
    return false;
  }
}
Matrix Matrix::T() const {
  Matrix res(shape()[1], shape()[0]);
  int idx_row, idx_col;
  for (idx_row = 0; idx_row < res.shape()[0]; ++idx_row) {
    for (idx_col = 0; idx_col < res.shape()[1]; ++idx_col) {
      res.mat[idx_row][idx_col] = mat[idx_col][idx_row];
    }
  }
  return res;
}
Matrix Matrix::reshape(int num_rows, int num_cols) const {
  if (num_rows * num_cols != shape()[0] * shape()[1]) {
    throw "Inconsistent shape";
  }
  std::vector<double> all_elements;
  for (Vector row_mat : mat) {
    int idx_col;
    for (idx_col = 0; idx_col < shape()[1]; ++idx_col) {
      all_elements.push_back(row_mat[idx_col]);
    }
  }
  Matrix res(num_rows, num_cols);
  int idx_row, idx_col, idx_all_ele = 0;
  for (idx_row = 0; idx_row < num_rows; ++idx_row) {
    for (idx_col = 0; idx_col < num_cols; ++idx_col) {
      res[idx_row][idx_col] = all_elements[idx_all_ele];
      ++idx_all_ele;
    }
  }
  return res;
}
Matrix Matrix::inverse() const {
// TODO
}
double Matrix::sum() const {
  double sum = 0;
  for (const Vector row : mat) {
    sum += row.sum();
  } 
  return sum;
}
Vector Matrix::sum(int dim) const {
  if (dim == 0) {
    Vector vec_sum(shape()[1]);
    int idx_row, idx_col;
    for (idx_row = 0; idx_row < shape()[0]; ++idx_row) {
      for (idx_col = 0; idx_col < shape()[1]; ++idx_col) {
        vec_sum[idx_col] += mat[idx_row][idx_col];
      }
    }
    return vec_sum;
  } else if (dim == 1) {
    Vector vec_sum(shape()[0]);
    int idx_row;
    for (idx_row = 0; idx_row < shape()[0]; ++idx_row) {
      vec_sum[idx_row] = mat[idx_row].sum();
    }
    return vec_sum;
  } else {
    throw "Invalid dimension";
  }
}
double Matrix::max() const {
  int idx_row;
  double max = mat[0].max();
  for (idx_row = 1; idx_row < shape()[0]; ++idx_row) {
    double row_max = mat[idx_row].max();
    if (row_max > max) {
      max = row_max;
    }
  }
  return max;
}
Vector Matrix::max(int dim) const {
  if (dim == 0) {
    Vector max_vec(shape()[1]);
    int idx_row, idx_col; 
    double max_of_col;
    for (idx_col = 0; idx_col < shape()[1]; ++idx_col) {
      max_of_col = mat[0][idx_col];
      for (idx_row = 1; idx_row < shape()[0]; ++idx_row) {
        if (mat[idx_row][idx_col] > max_of_col) {
          max_of_col = mat[idx_row][idx_col];
        }
      }
      max_vec[idx_col] = max_of_col;
    }
    return max_vec;
  } else if (dim == 1) {
    Vector max_vec(shape()[0]);
    int idx_row;
    for (idx_row = 0; idx_row < shape()[0]; ++idx_row) {
      max_vec[idx_row] = mat[idx_row].max();
    }
    return max_vec;
  } else {
    throw "Invalid dimension";
  }
}
double Matrix::min() const {
  int idx_row;
  double min = mat[0].min();
  for (idx_row = 1; idx_row < shape()[0]; ++idx_row) {
    double row_min = mat[idx_row].min();
    if (row_min > min) {
      min = row_min;
    }
  }
  return min;
}
Vector Matrix::min(int dim) const {
  if (dim == 0) {
    Vector min_vec(shape()[1]);
    int idx_row, idx_col; 
    double min_of_col;
    for (idx_col = 0; idx_col < shape()[1]; ++idx_col) {
      min_of_col = mat[0][idx_col];
      for (idx_row = 1; idx_row < shape()[0]; ++idx_row) {
        if (mat[idx_row][idx_col] > min_of_col) {
          min_of_col = mat[idx_row][idx_col];
        }
      }
      min_vec[idx_col] = min_of_col;
    }
    return min_vec;
  } else if (dim == 1) {
    Vector min_vec(shape()[0]);
    int idx_row;
    for (idx_row = 0; idx_row < shape()[0]; ++idx_row) {
      min_vec[idx_row] = mat[idx_row].min();
    }
    return min_vec;
  } else {
    throw "Invalid dimension";
  }
}
Matrix Matrix::cross(const Matrix & m) const {
  Matrix res = *this;
  int idx_row;
  for (idx_row = 0; idx_row < res.shape()[0]; ++idx_row) {
    res.mat[idx_row] *= m.mat[idx_row];
  }
  return res;
}
bool Matrix::approx(const Matrix & m, double error=1e-8) {
  if (shape()[0] != m.shape()[0] || 
      shape()[1] != m.shape()[1]) {
    return false;
  }
  int idx_row, idx_col;
  for (idx_row = 0; idx_row < shape()[0]; ++idx_row) {
    for (idx_col = 0; idx_col < shape()[1]; ++idx_col) {
      double actual_err = mat[idx_row][idx_col] - m.mat[idx_row][idx_col];
      actual_err = actual_err >= 0 ? actual_err : (-1) * actual_err;
      if (actual_err > error) {
        return false;
      }
    }
  }
  return true;
}
Matrix Matrix::approx(double value, double error=1e-8) {
  Matrix res(shape()[0], shape()[1]);
  int idx_row, idx_col;
  for (idx_row = 0; idx_row < shape()[0]; ++idx_row) {
    for (idx_col = 0; idx_col < shape()[1]; ++idx_col) {
      double actual_err = mat[idx_row][idx_col];
      actual_err = actual_err > 0 ? actual_err : (-1) * actual_err;
      if (actual_err <= error) {
        res.mat[idx_row][idx_col] = 1;
      }
    }
  }
  return res;
}
Matrix operator+(const Matrix & m_1, const Matrix & m_2) {  
  if (m_1.shape()[0] == m_2.shape()[0] &&
      m_1.shape()[1] == m_2.shape()[1]) {
    Matrix res = m_1;
    const int size_omp = 13;
    int idx_row;
    if (res.shape()[0] < size_omp) {
      for (idx_row = 0; idx_row < res.shape()[0]; ++idx_row) {
        res.mat[idx_row] += m_2.mat[idx_row];
      }       
    } else {
      #pragma omp parallel shared(res, m_2) private(idx_row)
      {
        #pragma omp for schedule(auto)
        for (idx_row = 0; idx_row < res.shape()[0]; ++idx_row) {
          res.mat[idx_row] += m_2.mat[idx_row];
        }         
      }
    }   
    return res;
  } else if (m_1.shape()[0] == m_2.shape()[0] && 
             m_1.shape()[1] == 1) {
    Matrix res = m_2;
    const int size_omp = 13;
    int idx_row;
    if (res.shape()[0] < size_omp) {
      for (idx_row = 0; idx_row < res.shape()[0]; ++idx_row) {
        res.mat[idx_row] += m_1.mat[idx_row][0];
      }
    } else {
      #pragma omp parallel shared(res, m_1) private(idx_row)
      {
        #pragma omp for schedule(auto)
        for (idx_row = 0; idx_row < res.shape()[0]; ++idx_row) {
          res.mat[idx_row] += m_1.mat[idx_row][0];
        }        
      }      
    }
    return res;
  } else if (m_1.shape()[0] == m_2.shape()[0] &&
             m_2.shape()[1] == 1) {
    Matrix res = m_1;
    const int size_omp = 13;
    int idx_row;
    if (res.shape()[0] < size_omp) {
      for (idx_row = 0; idx_row < res.shape()[0]; ++idx_row) {
        res.mat[idx_row] += m_2.mat[idx_row][0];
      }      
    } else {
      #pragma omp parallel shared(res, m_2) private(idx_row)
      {
        #pragma omp for schedule(auto)
        for (idx_row = 0; idx_row < res.shape()[0]; ++idx_row) {
          res.mat[idx_row] += m_2.mat[idx_row][0];
        }        
      }
    }
    return res;
  } else if (m_1.shape()[1] == m_2.shape()[1] &&
             m_1.shape()[0] == 1) {
    Matrix res = m_2;
    const int size_omp = 13;    
    int idx_row;
    if (res.shape()[0] < size_omp) {
      for (idx_row = 0; idx_row < res.shape()[0]; ++idx_row) {
        res.mat[idx_row] += m_1.mat[0];
      }
    } else {
      #pragma omp parallel shared(res, m_1) private(idx_row)
      {
        #pragma omp for schedule(auto)
        for (idx_row = 0; idx_row < res.shape()[0]; ++idx_row) {
          res.mat[idx_row] += m_1.mat[0];
        }        
      }
    }
    return res;
  } else if (m_1.shape()[1] == m_2.shape()[1] &&
             m_2.shape()[0] == 1) {
    Matrix res = m_1;
    const int size_omp = 13;
    int idx_row;
    if (res.shape()[0] < size_omp) {
      for (idx_row = 0; idx_row < res.shape()[0]; ++idx_row) {
        res.mat[idx_row] += m_2.mat[0];
      }
    } else {
      #pragma omp parallel shared(res, m_2) private(idx_row)
      {
        #pragma omp for schedule(auto)
        for (idx_row = 0; idx_row < res.shape()[0]; ++idx_row) {
          res.mat[idx_row] += m_2.mat[0];
        }        
      }
    }
    return res;
  } else {
    throw "Inconsistent shape";
  }
}
Matrix operator+(const Matrix & m, double value) {
  Matrix res = m;
  const int size_omp = 13;
  if (res.shape()[0] < size_omp) {
    for (Vector & row : res.mat) {
      row += value;
    }
  } else {
    int idx_row;
    #pragma omp parallel shared(res, value) private(idx_row)
    {
      #pragma omp for schedule(auto)
      for (idx_row = 0; idx_row < res.shape()[0]; ++idx_row) {
        res.mat[idx_row] += value;
      }
    }
  }
  return res;
}
Matrix operator+(double value, const Matrix & m) {
  return operator+(m , value);
}
Matrix operator-(const Matrix & m_1, const Matrix & m_2) {
  return m_1 + (-1) * m_2;
}
Matrix operator-(const Matrix & m, double value) {
  return m + (-1) * value;
}
Matrix operator-(double value, const Matrix & m) {
  return value + (-1) * m;
}
Matrix operator*(const Matrix & m_1, const Matrix & m_2) {
  if (m_1.shape()[1] != m_2.shape()[0]) {
    throw "Inconsistent shape";
  }
  Matrix res(m_1.shape()[0], m_2.shape()[1]);
  const int size_omp = 4;
  int idx_row, idx_col, idx_com;
  if ((res.shape()[0] + res.shape()[1]) / 2.0 < 4) {
    for (idx_row = 0; idx_row < res.shape()[0]; ++idx_row) {
      for (idx_col = 0; idx_col < res.shape()[1]; ++idx_col) {
        for (idx_com = 0; idx_com < m_1.shape()[1]; ++idx_com) {
          res.mat[idx_row][idx_col] +=
            m_1.mat[idx_row][idx_com] * m_2.mat[idx_com][idx_col];
        }
      }
    }
  } else {
    #pragma omp parallel shared(res, m_1, m_2) private(idx_row, idx_col, idx_com)
    {
      #pragma omp for schedule(auto) collapse(3)
      for (idx_row = 0; idx_row < res.shape()[0]; ++idx_row) {
        for (idx_col = 0; idx_col < res.shape()[1]; ++idx_col) {
          for (idx_com = 0; idx_com < m_1.shape()[1]; ++idx_com) {
            res.mat[idx_row][idx_col] +=
              m_1.mat[idx_row][idx_com] * m_2.mat[idx_com][idx_col];
          }
        }
      }      
    }
  }
  return res;
}
Matrix operator*(const Matrix & m, double value) {
  Matrix res = m;
  const int size_omp = 13;
  if (res.shape()[0] < size_omp) {
    for (Vector & row : res.mat) {
      row *= value;
    }    
  } else {
    int idx_row;
    #pragma omp parallel shared(res, value) private(idx_row)
    {
      #pragma omp for schedule(auto)
      for (idx_row = 0; idx_row < res.shape()[0]; ++idx_row) {
        res.mat[idx_row] *= value;
      }
    }
  }
  return res;
}
Matrix operator*(double value, const Matrix & m) {
  return operator*(m, value);
}
Matrix operator*(const Matrix & m, const Vector & v) {
  if (m.shape()[1] != v.shape()[0]) {
    throw "Inconsistent shape";
  }
  Matrix m_vec = v;
  Matrix res = m * m_vec;
  return res;
}
Matrix operator/(const Matrix & m_1, const Matrix & m_2) {
  if (m_1.shape()[0] != m_2.shape()[0] || 
      m_1.shape()[1] != m_2.shape()[1]) {
    throw "Inconsistent shape";
  }
  Matrix res = m_1;
  const int size_omp = 13;
  int idx_row;
  if (res.shape()[0] < size_omp) {
    for (idx_row = 0; idx_row < res.shape()[0]; ++idx_row) {
      res.mat[idx_row] = res.mat[idx_row] / m_2.mat[idx_row];
    }
  } else {
    #pragma omp parallel shared(res, m_2) private(idx_row)
    {
      #pragma omp for schedule(auto)
      for (idx_row = 0; idx_row < res.shape()[0]; ++idx_row) {
        res.mat[idx_row] = res.mat[idx_row] / m_2.mat[idx_row];
      }      
    }
  }
  return res;
}
Matrix operator/(const Matrix & m, double value) {
  return m * (1.0 / value);
}
Matrix operator/(double value, const Matrix & m) {
  Matrix res = m;
  const int size_omp = 13;
  if (res.shape()[0] < size_omp) {
    for (Vector & row : res.mat) {
      row = static_cast<double>(value) / row;
    }    
  } else {
    int idx_row;
    #pragma omp parallel shared(res, value) private(idx_row)
    {
      #pragma omp for schedule(auto)
      for (idx_row = 0; idx_row < res.shape()[0]; ++idx_row) {
        res[idx_row] = static_cast<double>(value) / res[idx_row];
      }
    }
  }
  return res;
}
void Matrix::operator+=(double value) {
  (*this) = (*this) + value;
}
void Matrix::operator+=(const Matrix & m) {
  (*this) = (*this) + m;
}
void Matrix::operator-=(double value) {
  (*this) = (*this) - value; 
}
void Matrix::operator-=(const Matrix & m) {
  (*this) = (*this) + m;
}
void Matrix::operator*=(double value) {
  (*this) = (*this) * value;
}
void Matrix::operator*=(const Matrix & m) {
  (*this) = (*this) * m;
}
void Matrix::operator/=(double value) {
  (*this) = (*this) / value;
}
void Matrix::operator/=(const Matrix & m) {
  (*this) = (*this) / m;
}
bool operator==(const Matrix & m_1, const Matrix & m_2) {
  if (m_1.shape()[0] != m_2.shape()[0] || 
      m_1.shape()[1] != m_2.shape()[1]) {
    return false;
  }
  int idx_row;
  for (idx_row = 0; idx_row < m_1.shape()[0]; ++idx_row) {
    if (m_1.mat[idx_row] != m_2.mat[idx_row]) {
      return false;
    }
  }
  return true;
}
Matrix operator==(const Matrix & m, double value) {
  Matrix res(m.shape()[0], m.shape()[1]);
  const int size_omp = 11;
  int idx_row;
  if (res.shape()[0] < size_omp) {
    for (idx_row = 0; idx_row < m.shape()[0]; ++idx_row) {
      res[idx_row] = (m[idx_row] == value);
    }
  } else {
    #pragma omp parallel shared(res, m, value) private(idx_row)
    {
      #pragma omp for schedule(auto)
      for (idx_row = 0; idx_row < m.shape()[0]; ++idx_row) {
        res[idx_row] = (m[idx_row] == value);
      }      
    }
  }
  return res;
}
Matrix operator==(double value, const Matrix & m) {
  return m == value;
}
bool operator!=(const Matrix & m_1, const Matrix & m_2) {
  return !(m_1 == m_2);
}
Matrix operator!=(const Matrix & m, double value) {
  Matrix res(m.shape()[0], m.shape()[1]);
  const int size_omp = 11;
  int idx_row;
  if (res.shape()[0] < size_omp) {
    for (idx_row = 0; idx_row < m.shape()[0]; ++idx_row) {
      res[idx_row] = (m[idx_row] != value);
    }
  } else {
    #pragma omp parallel shared(res, m, value) private(idx_row)
    {
      #pragma omp for schedule(auto)
      for (idx_row = 0; idx_row < m.shape()[0]; ++idx_row) {
        res[idx_row] = (m[idx_row] != value);
      }      
    }
  }
  return res;
}
Matrix operator!=(double value, const Matrix & m) {
  return m != value;
}
Matrix operator<(const Matrix & m_1, const Matrix & m_2) {
  if (m_1.shape()[0] != m_2.shape()[0] ||
      m_1.shape()[1] != m_2.shape()[1]) {
    throw "Inconsistent shape";
  }
  Matrix res(m_1.shape()[0], m_2.shape()[1]);
  const int size_omp = 11;
  int idx_row;
  if (res.shape()[0] < size_omp) {
    for (idx_row = 0; idx_row < res.shape()[0]; ++idx_row) {
      res[idx_row] = (m_1[idx_row] < m_2[idx_row]);
    }
  } else {
    #pragma omp parallel shared(res, m_1, m_2) private(idx_row)
    {
      #pragma omp for schedule(auto)
      for (idx_row = 0; idx_row < res.shape()[0]; ++idx_row) {
        res[idx_row] = (m_1[idx_row] < m_2[idx_row]);
      }      
    }
  }
  return res;
}
Matrix operator<(const Matrix & m, double value) {
  Matrix res(m.shape()[0], m.shape()[1]);
  const int size_omp = 11;
  int idx_row;
  if (res.shape()[0] < size_omp) {
    for (idx_row = 0; idx_row < m.shape()[0]; ++idx_row) {
      res[idx_row] = (m[idx_row] < value);
    }
  } else {
    #pragma omp parallel shared(res, m, value) private(idx_row)
    {
      #pragma omp for schedule(auto)
      for (idx_row = 0; idx_row < m.shape()[0]; ++idx_row) {
        res[idx_row] = (m[idx_row] < value);
      }      
    }
  }
  return res;
}
Matrix operator<(double value, const Matrix & m) {
  Matrix res(m.shape()[0], m.shape()[1]);
  const int size_omp = 11;
  int idx_row;
  if (res.shape()[0] < size_omp) {
    for (idx_row = 0; idx_row < m.shape()[0]; ++idx_row) {
      res[idx_row] = (value < m[idx_row]);
    }
  } else {
    #pragma omp parallel shared(res, m, value) private(idx_row)
    {
      #pragma omp for schedule(auto)
      for (idx_row = 0; idx_row < m.shape()[0]; ++idx_row) {
        res[idx_row] = (value < m[idx_row]);
      }      
    }
  }
  return res;
}
Matrix operator<=(const Matrix & m_1, const Matrix & m_2) {
  if (m_1.shape()[0] != m_2.shape()[0] ||
      m_1.shape()[1] != m_2.shape()[1]) {
    throw "Inconsistent shape";
  }
  Matrix res(m_1.shape()[0], m_2.shape()[1]);
  const int size_omp = 11;
  int idx_row;
  if (res.shape()[0] < size_omp) {
    for (idx_row = 0; idx_row < res.shape()[0]; ++idx_row) {
      res[idx_row] = (m_1[idx_row] <= m_2[idx_row]);
    }
  } else {
    #pragma omp parallel shared(res, m_1, m_2) private(idx_row)
    {
      #pragma omp for schedule(auto)
      for (idx_row = 0; idx_row < res.shape()[0]; ++idx_row) {
        res[idx_row] = (m_1[idx_row] <= m_2[idx_row]);
      }      
    }
  }
  return res;
}
Matrix operator<=(const Matrix & m, double value) {
  Matrix res(m.shape()[0], m.shape()[1]);
  const int size_omp = 11;
  int idx_row;
  if (res.shape()[0] < size_omp) {
    for (idx_row = 0; idx_row < m.shape()[0]; ++idx_row) {
      res[idx_row] = (m[idx_row] <= value);
    }
  } else {
    #pragma omp parallel shared(res, m, value) private(idx_row)
    {
      #pragma omp for schedule(auto)
      for (idx_row = 0; idx_row < m.shape()[0]; ++idx_row) {
        res[idx_row] = (m[idx_row] <= value);
      }      
    }
  }
  return res;
}
Matrix operator<=(double value, const Matrix & m) {
  Matrix res(m.shape()[0], m.shape()[1]);
  const int size_omp = 11;
  int idx_row;
  if (res.shape()[0] < size_omp) {
    for (idx_row = 0; idx_row < m.shape()[0]; ++idx_row) {
      res[idx_row] = (value <= m[idx_row]);
    }
  } else {
    #pragma omp parallel shared(res, m, value) private(idx_row)
    {
      #pragma omp for schedule(auto)
      for (idx_row = 0; idx_row < m.shape()[0]; ++idx_row) {
        res[idx_row] = (value <= m[idx_row]);
      }      
    }
  }
  return res;
}
Matrix operator>(const Matrix & m_1, const Matrix & m_2) {
  return m_2 < m_1;
}
Matrix operator>(const Matrix & m, double value) {
  return value < m;
}
Matrix operator>(double value, const Matrix & m) {
  return m < value;
}
Matrix operator>=(const Matrix & m_1, const Matrix & m_2) {
  return m_2 <= m_1;
}
Matrix operator>=(const Matrix & m, double value) {
  return value <= m;
}
Matrix operator>=(double value, const Matrix & m) {
  return m <= value;
}
Vector & Matrix::operator[](int index) {
  if (index >= 0) {
    if (index >= shape()[0]) {
      throw "Out-of-range";
    }
    return mat.at(index);
  } else if (index < 0) {
    if (index > shape()[0]) {
      throw "Out-of-range";
    }
    return mat.at(mat.size() - index);
  }
}
const Vector & Matrix::operator[](int index) const {
  if (index >= 0) {
    if (index >= shape()[0]) {
      throw "Out-of-range";
    }
    return mat.at(index);
  } else if (index < 0) {
    if (index > shape()[0]) {
      throw "Out-of-range";
    }
    return mat.at(mat.size() - index);
  }
}
Matrix Matrix::operator()(int idx_row) const {
  Matrix res(1, shape()[1]);
  res.mat[0] = mat[idx_row];
  return res;
}
Matrix Matrix::operator()(int r_begin, int r_end) const {
  if (r_begin > r_end) {
    int tmp_swap = r_begin;
    r_begin = r_end;
    r_end = tmp_swap;
  }
  if (r_end >= shape()[0]) {
    r_end = shape()[0];
  }
  Matrix res(r_end - r_begin, shape()[1]);
  int idx_row;
  for (idx_row = 0; idx_row < r_end - r_begin; ++idx_row) {
    res.mat[idx_row] = mat[r_begin + idx_row];
  }
  return res;
}
Matrix Matrix::operator()(int r_begin, int r_end, 
                          int c_begin, int c_end) const {
  if (r_begin > r_end) {
    int tmp_swap = r_begin;
    r_begin = r_end;
    r_end = tmp_swap;
  }
  if (c_begin > c_end) {
    int tmp_swap = c_begin;
    c_begin = c_end;
    c_end = tmp_swap;
  }
  if (r_end > shape()[0] || c_end > shape()[1]) {
    throw "Out-of-range";
  }
  Matrix res(r_end - r_begin, c_end - c_begin);
  int idx_row;
  for (idx_row = 0; idx_row < (r_end - r_begin); ++idx_row) {
    res.mat[idx_row] = mat[r_begin + idx_row](c_begin, c_end);
  }
  return res;
}
std::ostream & operator<<(std::ostream & out, const Matrix & m) {
  out << "[";
  int idx_row = 0;
  for (const Vector row : m.mat) {
    out << row;
    if (idx_row != m.shape()[0] - 1) {
      out << "\n ";
    }
    ++idx_row;
  }
  out << "]";
  return out;
}
std::istream & operator>>(std::istream & in, Matrix & m) {
  for (Vector & row : m.mat) {
    in >> row;
  }
  return in;
}
