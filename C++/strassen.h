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


This is a implementation of the Strassen algorithm.

Created on 2016-07-26 by WaizungTaam
Email: waizungtaam@gmail.com

Reference: https://en.wikipedia.org/wiki/Strassen_algorithm

*/

#ifndef STRASSEN_H
#define STRASSEN_H

#include <cmath>
#include <vector>
#include "matrix/matrix.h"

namespace strassen {

Matrix _pad(const Matrix &, int);
std::vector<Matrix> _partition(const Matrix &);
Matrix _mat_mul_2x2(const Matrix &, const Matrix &);
Matrix _iterate(const Matrix &, const Matrix &);

Matrix strassen(const Matrix & X, const Matrix & Y) {
  if (X.shape()[1] != Y.shape()[0]) {
    throw "Inconsistent shape for strassen()";
  }
  int max_size = std::fmax(std::fmax(X.shape()[0], X.shape()[1]), 
                           std::fmax(Y.shape()[0], Y.shape()[1]));
  if (max_size == 0) {
    return Matrix(0, 0);
  } else if (max_size == 1) {
    return Matrix(1, 1, X[0][0] * Y[0][0]);
  }
  int square_size = std::exp2(std::ceil(std::log2(max_size)));

  Matrix A = _pad(X, square_size);
  Matrix B = _pad(Y, square_size);

  Matrix C = _iterate(A, B);

  return C(0, X.shape()[0], 0, Y.shape()[1]);
}


Matrix _pad(const Matrix & M, int size) {
  Matrix M_pad(size, size, 0);
  M_pad = M_pad.replace(M, 0, 0);
  return M_pad;
}

std::vector<Matrix> _partition(const Matrix & M) {
  std::vector<Matrix> M_s(4);
  int block_size = M.shape()[0] / 2;
  M_s[0] = M(0, block_size, 0, block_size);
  M_s[1] = M(0, block_size, block_size, block_size * 2);
  M_s[2] = M(block_size, block_size * 2, 0, block_size);
  M_s[3] = M(block_size, block_size * 2, block_size, block_size * 2);
  return M_s;
}

Matrix _mat_mul_2x2(const Matrix & M, const Matrix & N) {
  return Matrix({{M[0][0] * N[0][0] + M[0][1] * N[1][0], 
                  M[0][0] * N[0][1] + M[0][1] * N[1][1]}, 
                 {M[1][0] * N[0][0] + M[1][1] * N[1][0],
                  M[1][0] * N[0][1] + M[1][1] * N[1][1]}});
}

Matrix _iterate(const Matrix & A, const Matrix & B) {
  if (A.shape()[0] == 2 && A.shape()[1] == 2 && 
      B.shape()[0] == 2 && B.shape()[1] == 2) {
    return _mat_mul_2x2(A, B);
  }

  std::vector<Matrix> A_s = _partition(A);
  std::vector<Matrix> B_s = _partition(B);
  Matrix A_11 = A_s[0];  Matrix A_12 = A_s[1]; 
  Matrix A_21 = A_s[2];  Matrix A_22 = A_s[3];
  Matrix B_11 = B_s[0];  Matrix B_12 = B_s[1]; 
  Matrix B_21 = B_s[2];  Matrix B_22 = B_s[3];
  A_s.clear();  B_s.clear();

  Matrix M_1 = _iterate(A_11 + A_22, B_11 + B_22);
  Matrix M_2 = _iterate(A_21 + A_22, B_11);
  Matrix M_3 = _iterate(A_11, B_12 - B_22);
  Matrix M_4 = _iterate(A_22, B_21 - B_11);
  Matrix M_5 = _iterate(A_11 + A_12, B_22);
  Matrix M_6 = _iterate(A_21 - A_11, B_11 + B_12);
  Matrix M_7 = _iterate(A_12 - A_22, B_21 + B_22);

  Matrix C(A.shape());
  int block_size = C.shape()[0] / 2;
  C = C.replace(M_1 + M_4 - M_5 + M_7, 0, 0);
  C = C.replace(M_3 + M_5, 0, block_size);
  C = C.replace(M_2 + M_4, block_size, 0);
  C = C.replace(M_1 - M_2 + M_3 + M_6, block_size, block_size);

  return C;
}

}  // namespace strassen

#endif  // strassen.h