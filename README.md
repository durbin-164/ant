# ant

[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=durbin-164_ant&metric=alert_status)](https://sonarcloud.io/dashboard?id=durbin-164_ant)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=durbin-164_ant&metric=coverage)](https://sonarcloud.io/dashboard?id=durbin-164_ant)
[![Code Smells](https://sonarcloud.io/api/project_badges/measure?project=durbin-164_ant&metric=code_smells)](https://sonarcloud.io/dashboard?id=durbin-164_ant)
[![Security Rating](https://sonarcloud.io/api/project_badges/measure?project=durbin-164_ant&metric=security_rating)](https://sonarcloud.io/dashboard?id=durbin-164_ant)
[![Vulnerabilities](https://sonarcloud.io/api/project_badges/measure?project=durbin-164_ant&metric=vulnerabilities)](https://sonarcloud.io/dashboard?id=durbin-164_ant)
[![Bugs](https://sonarcloud.io/api/project_badges/measure?project=durbin-164_ant&metric=bugs)](https://sonarcloud.io/dashboard?id=durbin-164_ant)


## About 

ANT is a NdArray that is developing with c++ and Cuda from scratch. Currently, it supports some basic array operations just like NumPy or Cupy but in language, C++ instate of python. It is an educational project to understand how a NumPy-like library works, and how automatic broadcasting work. Apart from this project also helps to learn Cuda programming and how our GPU work to do parallel programming. To learn how this ant array work visits the **documentation and test folder.**


## API Documentations: [here](https://durbin-164.github.io/ant/)

## Some functionalities of ant NdArray

### Array Creation
```cpp
#include "array.h"

ndarray::Shape s = {3};
double a[] = {1,2,3};
ndarray::Array A(s, a);

EXPECT_EQ(A.rank(),1);
EXPECT_EQ(A.size(), 3);
```

### Addition
```cpp
ndarray::Shape s = {3};
double a[] = {1,2,3};

ndarray::Array A(s, a);
ndarray::Array B(s, a);

auto C = A + B;


EXPECT_EQ(A.rank(),1);
EXPECT_EQ(B.rank(), 1);
EXPECT_EQ(C.rank(), 1);
EXPECT_EQ(C.size(), 3);

double * cActualData = C.hostData();
double cExpectedData[] = {2,4,6};
for(int i =0; i< C.size(); i++){
    EXPECT_EQ(cActualData[i], cExpectedData[i]);
}
```

### Broadcasting
```cpp
ndarray::Shape a_shape = {4,1};
ndarray::Shape b_shape = {1,3};

double a_data[] ={1,2,3,4};
double b_data[] = {1,2,3};
ndarray::Array A(a_shape, a_data);
ndarray::Array B(b_shape, b_data);

auto C = A+B;

double *actual = C.hostData();
double expected[] = {2,3,4,3,4,5,4,5,6,5,6,7};

VectorEQ(C.shape(), {4,3});
DoubleArrayEQ(actual, expected, 12); 
```

### Matmul
```cpp
 ndarray::Shape a_shape = {2,4};
ndarray::Shape b_shape = {4,2};

double a_data[] = {1,2,3,4,5,6,7,8};
double b_data[] ={8,7,6,5,4,3,2,1};
ndarray::Array A(a_shape, a_data);
ndarray::Array B(b_shape, b_data);

auto C = A.matmul(B);

double *actual = C.hostData();
double expected[] = {40,30,120,94};
VectorEQ(C.shape(), {2,2});
DoubleArrayEQ(actual, expected, 4); 
```

### Matmul with broadcasting
```cpp
 ndarray::Shape a_shape = {2,2};
ndarray::Shape b_shape = {2,2,3};

double a_data[] = {1,2,3,4};
double b_data[] ={12,11,10,9,8,7,6,5,4,3,2,1};
ndarray::Array A(a_shape, a_data);
ndarray::Array B(b_shape, b_data);

auto C = A.matmul(B);

double *actual = C.hostData();
double expected[] = {30,27,24,72,65,58, 12,9,6,30,23,16};

VectorEQ(C.shape(), {2,2,3});
DoubleArrayEQ(actual, expected, 12); 
```

### Transpose
```cpp
ndarray::Shape a_shape = {5,4};
double a_data[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};

ndarray::Array A(a_shape, a_data);
A.transpose({1,0});

double expected[][5] = {{1,5,9,13,17},{2,6,10,14,18},{3,7,11,15,19},{4,8,12,16,20}};
VectorEQ(A.shape(), {4,5});
VectorEQ(A.stride(), {1,4});
```

### Indexing
```cpp
ndarray::Shape shape= {2,3};
double data[][3] = {{1,2,3},{10,20,30}};
ndarray::Array A(shape, *data);

EXPECT_EQ(A(0,0), 1);
EXPECT_EQ(A(0,1), 2);
EXPECT_EQ(A(0,2), 3);

EXPECT_EQ(A(1,0), 10);
EXPECT_EQ(A(1,1), 20);
EXPECT_EQ(A(1,2), 30);
```

### Slicing
```cpp
ndarray::Shape a_shape = {3,3,2};
double a_data[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18};

ndarray::Array A(a_shape, a_data);
ndarray::Array B = A[{{1,5,1}, {0,-1},{1}}];

double *actual = B.hostData();
double expected[] = {8,10,14,16};
VectorEQ(B.shape(), {2,2});
DoubleArrayEQ(actual, expected, 4); 
```

### Details example are found in the test folder, where different kind of scenerio was tasted. 