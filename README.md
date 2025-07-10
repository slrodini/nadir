# Nadir

`nadir` is a minimization library that includes a family of variants of the Adam base algorithm
and in the future will include a number of global, non-gradient based, minimization algorithm.

## Installation
**Prerequisite:** 
1. You should have installed a C++ compiler with support for C++20.
2. You should have installed `cmake`
3. You should have install the [Eigen library](https://eigen.tuxfamily.org/), version >= 3.3. 
   1. Note *i)* If you install Eigen in a directory not searched by default, you can instruct `cmake` to where to look for Eigen via `-DEigen3_DIR=<Eigen-install-prefix>/share/eigen3/cmake`.
   2. Note *ii)* If you are working with the [Oros library](https://github.com/MapCollaboration/Oros), you can use the Eigen distribution that comes with it. In that case, please use `-DEigen3_DIR=<Oros-install-prefix>/share/Oros/eigen3/cmake` when running `cmake`.

**Installation:**
The usual `cmake` procedure. From within `nadir` directory
```shell
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/your/installation/path/ ..
make
make install
```
*Note:* By the default, if no prefix specification is given, `nadir` will be installed in the `/usr/local/`  directory. If you want (or need) to use a different path, remember to export the `<install-prefix>/lib` folder into the `LD_LIBRARY_PATH` and the `<install-prefix>/bin` folder int the `PATH`.

After installation, you can run 
```shell
Nadir-config --help
```
to get the list of available flags to be used in your project when using `nadir`.

The library can be uninstalled by running:

```bash
make clean
xargs rm < install_manifest.txt
```
