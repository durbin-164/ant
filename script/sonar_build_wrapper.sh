wget -q https://sonarcloud.io/static/cpp/build-wrapper-linux-x86.zip
unzip -q build-wrapper-linux-x86.zip
rm -r build-wrapper-linux-x86.zip

./build-wrapper-linux-x86/build-wrapper-linux-x86-64 --out-dir bw-output python setup.py build_ext

# rm -r build-wrapper-linux-x86