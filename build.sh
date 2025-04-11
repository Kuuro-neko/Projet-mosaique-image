#!/bin/sh
mkdir -p build
cd build
cmake ..
make

# ask if launch the program (y/n)

echo "Do you also want to launch the program? (y/n)"
read answer
if [ "$answer" != "${answer#[Yy]}" ]; then
    cd ..
    ./build/main
fi
