/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/
#pragma once
//NOTE: this header can be removed after paddle's extension apply C++17. Then, we can simply use std::optional

#if __cplusplus >= 201703L
#  include <optional>
#elif __cplusplus >= 201402L
#  include <experimental/optional>
#  ifndef PYBIND11_HAS_EXP_OPTIONAL
     //paddle's dependency define <optional> in c++17, which confuse pybind11 from correctly defining PYBIND11_HAS_EXP_OPTIONAL. Here forcely define it
#    define PYBIND11_HAS_EXP_OPTIONAL 1
#  endif
//pybind will use <experimental/optional> of c++14 if PYBIND11_HAS_EXP_OPTIONAL is defined
#  include <pybind11/stl.h>  
#  define EXP_OPTIONAL_OF_TENSOR
#else
#  error "__cplusplus is undefined!"
#endif
