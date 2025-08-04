- get_L_N.py tells you what you can send as inputs
- main_maze_md.py is the main file to run with inputs according to those templates in the above file
- run ./compile_c.sh to compile the C functions

## Compile using CMake

Needs cmake version 3.20 or higher to be installed.
Starting from the root directory of the project, run the following commands:

```bash

cd library

rm -rf build  # Remove the build directory if it exists

cmake -S . -B build -DCMAKE_INSTALL_PREFIX=../maze_poisson  # Install the library next to the python files
cmake --build build  # Build the library
cmake --install build  # Install the library

```

if from macOS:

```bash

cd library

rm -rf build

cmake -S . -B build \
  -DCMAKE_INSTALL_PREFIX=../maze_poisson \
  -DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm/bin/clang

cmake --build build -j8
cmake --install build
```