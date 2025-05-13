#!/bin/sh

sudo apt-get update -y

sudo apt-get install build-essential -y

sudo apt-get install gcc-multilib -y 

mkdir gcc_install

cd gcc_install

wget http://mirrors.ustc.edu.cn/gnu/gcc/gcc-9.2.0/gcc-9.2.0.tar.gz

tar -xvf gcc-9.2.0.tar.gz

cd gcc-9.2.0

sed -i 's/ftp/http/g' contrib/download_prerequisites

sed -e '1161 s|^|//|'     -i libsanitizer/sanitizer_common/sanitizer_platform_limits_posix.cc

./contrib/download_prerequisites

cd ..

mkdir gcc-build

cd gcc-build

../gcc-9.2.0/configure --disable-multilib --enable-language=c,c++ --prefix=$(pwd)/build

sudo make -j$(nproc)

sudo make install