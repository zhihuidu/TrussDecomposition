# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash -e

if [ ! -f pkxsort.h ]
then
  wget https://raw.githubusercontent.com/voutcn/kxsort/778721d9b44bb942c0efbe9699379d4004c40298/kxsort.h -O pkxsort.h
  sha256sum -c sha256sums || exit 1
  patch pkxsort.h < pkxsort.h.patch
fi

#CXX=clang++
CXX=g++
#CXXFLAGS="-O3 -g -fopenmp -Wall -march=native -I $HOME/anaconda3/include -I $HOME/anaconda3/include/gflags -I /rhome/zhihui/anaconda3/lib/clang/10.0.1/include -L $HOME/anaconda3/lib -L $HOME/anaconda3/lib/clang/"
CXXFLAGS="-O3 -g -fopenmp -Wall -march=native -I $HOME/anaconda3/include -L $HOME/anaconda3/lib  -lgflags "
${CXX} minsearch.cc ${CXXFLAGS} -o minsearch -L$HOME/anoconda3/lib/libgflags.so.2.2.2
