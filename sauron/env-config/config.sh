#!/bin/bash

yum install gcc gcc-c++ cmake qt5-qtbase-devel -y
cd opencv*
mkdir build
cd biuld
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j4
make install
cd ../..

git clone https://github.com/biometrics/openbr.git
cd openbr
git checkout 0.5
git submodule init
git submodule update

# $Id: $


