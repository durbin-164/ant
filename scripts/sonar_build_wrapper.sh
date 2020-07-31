mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug

wget -q https://sonarcloud.io/static/cpp/build-wrapper-linux-x86.zip
unzip -q build-wrapper-linux-x86.zip
rm -r build-wrapper-linux-x86.zip



./build-wrapper-linux-x86/build-wrapper-linux-x86-64 --out-dir bw_output make clean all

# rm -r build-wrapper-linux-x86