"""Copyright 2016 WaizungTaam.  All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


This is a test of the Strassen algorithm impementation.

Created on 2016-07-25 by WaizungTaam
Email: waizungtaam@gmail.com

Reference: https://en.wikipedia.org/wiki/Strassen_algorithm

"""

import numpy as np
import time

import strassen


def mat_mul(X, Y):
    """Common matrix multiplication algorithm O(n^3)
    """
    if X.shape[1] != Y.shape[0]:
        raise Exception("Inconsistent shape for matrix multiplication.")
    Z = np.zeros((X.shape[0], Y.shape[1]))
    for i in range(X.shape[0]):
        for j in range(Y.shape[1]):
            for k in range(X.shape[1]):
                Z[i][j] += X[i][k] * Y[k][j]
    return Z

def test_validation():
    """A simple test for the validation of the algorithm.
    """
    X = np.array([[1, 2, 3, 4, 5], 
                  [2, 3, 4, 5, 6], 
                  [3, 4, 5, 6, 7]])
    Y = np.array([[1, 3, 5, 7], 
                  [2, 4, 6, 8], 
                  [3, 5, 7, 9], 
                  [4, 6, 8, 10], 
                  [5, 7, 9, 11]])

    print(strassen.strassen(X, Y))
    print(np.dot(X, Y))

def test_time_algorithm():
    """A test that compare the time cost of different algorithms.
    """
    size = 256
    X = np.random.rand(size, size)
    Y = np.random.rand(size, size)

    t_strassen_begin = time.clock()
    strassen.strassen(X, Y)
    t_strassen_end = time.clock()
    print(t_strassen_end - t_strassen_begin)

    t_mat_mul_begin = time.clock()
    mat_mul(X, Y)
    t_mat_mul_end = time.clock()
    print(t_mat_mul_end - t_mat_mul_begin)

    t_np_begin = time.clock()
    np.dot(X, Y)
    t_np_end = time.clock()
    print(t_np_end - t_np_begin)

def test_time_size():
    """A test that compares the time cost changes of Strassen Algorithm and 
       common multiplication as the sizes of matrices increase.
    """
    for size in range(300):
        X = np.random.rand(size, size)
        Y = np.random.rand(size, size)

        t_strassen_begin = time.clock()
        strassen.strassen(X, Y)
        t_strassen_end = time.clock()
        t_strassen = t_strassen_end - t_strassen_begin     

        t_mat_mul_begin = time.clock()
        mat_mul(X, Y)
        t_mat_mul_end = time.clock()
        t_mat_mul = t_mat_mul_end - t_mat_mul_begin

        print(size, t_strassen, t_mat_mul)



if __name__ == "__main__":
    test_validation()
    test_time_algorithm()
    test_time_size()