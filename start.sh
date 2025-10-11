CC=/opt/rocm/bin/amdclang CXX=/opt/rocm/bin/amdclang++ \
cmake -B build . \
  -DROCWMMA_BUILD_TESTS=OFF \
  -DROCWMMA_BUILD_SAMPLES=ON \
  -DGPU_TARGETS=gfx942 \
  -DCMAKE_BUILD_TYPE=Debug
