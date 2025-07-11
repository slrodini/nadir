# Nadir

`nadir` is a minimization library that includes Adam optimizer and many of its variants and a fully customizable Simulated Annealing algorithm.

> The library is still very much work-in-progress. The API is somewhat stable, but major changes can still happen.

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

## Algorithms

`nadir` provides a number of Adam derived algorithm, as well as a customizable implementation of the Simulated Annealing

#### Adam and variants
The implemented variants are:
- The classical Adam https://arxiv.org/pdf/1412.6980
- The AMSGrad improved version https://arxiv.org/pdf/1904.09237, https://arxiv.org/pdf/1904.03590 with the bias correction applied before the max (AMSGRAD) or after the max (AMSGRAD_V2)
- The Nesterov-accellerated Adam https://openreview.net/pdf/OM0jvwB8jIp57ZJjtNEZ.pdf
- The AdamW variant (decoupled weight decay) https://arxiv.org/pdf/1711.05101
- The AdaBelief variant https://arxiv.org/pdf/2010.07468 without decouple weight decay (ADABELIEF) and with decoupled weight decay (ADABELIEF_W)
- The evoLved sIgn mOMentum (LION) pseudo variant https://arxiv.org/pdf/2302.06675
- The Rectified Adam (RAdam) variant https://arxiv.org/pdf/1908.03265

More variants are likely to come in the future!

#### Simulated Anneling

The Simulated Annealing (SA) implementation is based on the component-based analysis of
Computers and Operations Research 104 (2019) 191–206 (https://doi.org/10.1016/j.cor.2018.12.015).
All the components are customizable (i.e. the user can define its own implementation for them, given a fixed interface for each). `nadir` comes with most of the components defined in the paper pre-implemented, save few exceptions that either are not suited for the typical use case of `nadir` or that require too much meta-information to work and are currently not implemented (but this might change in the future, based on the practical requirements).
Moreover, I opted to not have a Temperature Restart component, in favor of having the possibility of chaining different minimizers, which allow essentially to have temperature restart externally. Since the context of one SA can be chained to the next, one can preserve the important information.

The six components are:
- ITx: Initial Temperature (and final one)
- SCx: Stopping criterion
- NEx: Exploration criterion (how a neighbor is selected)
- ACx: Acceptance criterion (how the current proposed solution is either accepted or rejected)
- CSx: Cooling scheme (how the temperature is decreased)
- TLx: Temperature Length (how many step at fix temperature are performed)

> Important: there is no fix upper bound to the computational time or number of iterations, unless a SCx that implements it is used. If you are unsure of your choices, please use a SCx with a fix upper bound of iterations or use the NadirIterCallback class to abort under your condition of choice.
