# ReSTIR and ReSTIR PT with lajolla
[ReSTIR](https://research.nvidia.com/labs/rtr/publication/bitterli2020spatiotemporal/) and [ReSTIR PT](https://research.nvidia.com/labs/rtr/publication/lin2022generalized/) implementation on [lajolla](https://github.com/BachiLi/lajolla_public) renderer

# Build
All the dependencies are included. Use CMake to build.
If you are on Unix systems, try
```
mkdir build
cd build
cmake ..
cmake --build .
```
It requires compilers that support C++17 (gcc version >= 8, clang version >= 7, Apple Clang version >= 11.0, MSVC version >= 19.14).

Apple M1 users: you might need to build Embree from scratch since the prebuilt MacOS binary provided is built for x86 machines. (But try build command above first.)

# Run
Try 
```
cd build
./lajolla ../scenes/cbox/cbox.xml
```
This will generate an image "image.exr".

To view the image, use [hdrview](https://github.com/wkjarosz/hdrview), or [tev](https://github.com/Tom94/tev).

# ReSTIR and ReSTIR PT
ReSTIR and ReSTIR PT are implemented with just spatial reuse, since lajolla is built to render images only.

Checkout scene file "scenes/restir_test_scene2/scene2.xml" for the parameters of ReSTIR and ReSTIR PT.

While ReSTIR is implemented exactly same as presented in the paper, ReSTIR PT is implemented with some specific choices and limitations, please take a look at the report file for more details.
