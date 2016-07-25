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

#include <iostream>
#include <chrono>
#include "matrix/matrix.h"
#include "strassen.h"

void test_validation() {
  Matrix X = {{1, 2, 3, 4, 5},
              {2, 3, 4, 5, 6},
              {3, 4, 5, 6, 7}},
         Y = {{1, 3, 5, 7},
              {2, 4, 6, 8},
              {3, 5, 7, 9},
              {4, 6, 8, 10},
              {5, 7, 9, 11}};
  std::cout << strassen::strassen(X, Y) << std::endl;
  std::cout << X * Y << std::endl;
}

void test_time_algorithm() {
  int size = 256;
  Matrix X(size, size, "uniform", -1.0, 1.0),
         Y(size, size, "uniform", -1.0, 1.0);

  std::chrono::time_point<std::chrono::steady_clock> t_strassen_begin =  
    std::chrono::steady_clock::now();
  strassen::strassen(X, Y);
  std::chrono::time_point<std::chrono::steady_clock> t_strassen_end =  
    std::chrono::steady_clock::now();
  std::chrono::duration<double> t_strassen_diff = 
    t_strassen_end - t_strassen_begin;    
  std::cout << t_strassen_diff.count() << std::endl;

  std::chrono::time_point<std::chrono::steady_clock> t_mat_mul_begin =  
    std::chrono::steady_clock::now();  
  X * Y;
  std::chrono::time_point<std::chrono::steady_clock> t_mat_mul_end =  
    std::chrono::steady_clock::now();
  std::chrono::duration<double> t_mat_mul_diff = 
    t_mat_mul_end - t_mat_mul_begin;    
  std::cout << t_mat_mul_diff.count() << std::endl;
}

void test_time_size() {
  int max_size = 300, size;
  for (size = 1; size < max_size; ++size) {
    Matrix X(size, size, "uniform", -1.0, 1.0);
    Matrix Y(size, size, "uniform", -1.0, 1.0);

    std::chrono::time_point<std::chrono::steady_clock> t_strassen_begin =  
      std::chrono::steady_clock::now();
    strassen::strassen(X, Y);
    std::chrono::time_point<std::chrono::steady_clock> t_strassen_end =  
      std::chrono::steady_clock::now();
    std::chrono::duration<double> t_strassen_diff = 
      t_strassen_end - t_strassen_begin;

    std::chrono::time_point<std::chrono::steady_clock> t_mat_mul_begin =  
      std::chrono::steady_clock::now();  
    X * Y;
    std::chrono::time_point<std::chrono::steady_clock> t_mat_mul_end =  
      std::chrono::steady_clock::now();
    std::chrono::duration<double> t_mat_mul_diff = 
      t_mat_mul_end - t_mat_mul_begin;

    std::cout << size << "\t" 
              << t_strassen_diff.count() << "\t"
              << t_mat_mul_diff.count() << "\n";
  }
}


int main() {
  test_validation();
  test_time_algorithm();
  test_time_size();
}