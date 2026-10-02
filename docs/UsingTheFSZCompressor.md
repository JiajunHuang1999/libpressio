# Using the FSZ Compressor {#usingfsz}

[FSZ](https://github.com/JiajunHuang1999/FSZ) is an extremely fast GPU error-bounded lossy compressor in the SZ family.
It stands at the family's Pareto frontier: achieving both extreme throughput and a high compression ratio simultaneously.
The `fsz` plugin exposes it through LibPressio.
FSZ is a CUDA library, so the plugin requires a CUDA-enabled build of LibPressio and a GPU with compute capability 8.0 or newer.

## Building

First build and install FSZ (CUDA 12.x or 13.x, CMake 3.18 or newer):

```bash
git clone https://github.com/JiajunHuang1999/FSZ.git
cmake -S FSZ -B FSZ/build -DCMAKE_INSTALL_PREFIX=$PREFIX
cmake --build FSZ/build -j
cmake --install FSZ/build
```

Then configure LibPressio with the CUDA domains and the FSZ plugin:

```bash
cmake -S libpressio -B build -DLIBPRESSIO_HAS_CUDA=ON -DLIBPRESSIO_HAS_FSZ=ON -DCMAKE_PREFIX_PATH=$PREFIX
cmake --build build -j
cmake --install build
```

`LIBPRESSIO_HAS_FSZ` requires `LIBPRESSIO_HAS_CUDA=ON` because the plugin moves data through the `cudamalloc` domain.
LibPressio links the static `fsz::fsz` target exported by FSZ's `find_package(fsz)` package.
Projects that consume the installed LibPressio with `find_package(LibPressio)` need the FSZ prefix in `CMAKE_PREFIX_PATH` as well.

## Testing

On a machine with a CUDA GPU:

```bash
ctest --test-dir build -R "FSZPlugin|fsz" --output-on-failure
```

`test/test_fsz_plugin.cc` checks round trips of float and double data from host memory against the error bound,
compression and decompression into caller-provided device buffers without copies,
byte equality between the plugin's stream and the stream from FSZ's own API,
a value-range relative bound through the `pressio` meta-compressor, and error reporting.
The generic `test_compressor_integration` suite also exercises `fsz` (documentation, configuration, and the shared data cases).

## Options and behavior

| option        | type   | meaning                                                   |
|---------------|--------|-----------------------------------------------------------|
| `pressio:abs` | double | absolute error bound, must be positive (default `1e-4`)  |

+ Supported types are `pressio_float_dtype` and `pressio_double_dtype`, with data of any dimensionality.
+ Inputs may be in host or device memory. Host inputs are copied to the GPU, and device inputs are used in place.
+ Compressed and decompressed outputs are returned in device memory (`cudamalloc` domain).
  A caller-provided device buffer is reused when it is large enough, so preallocated buffers avoid allocations.
  LibPressio's IO and metrics plugins copy device data to the host when they need it.
  In C++, `domain_manager().make_readable(domain_plugins().build("malloc"), data)` does the same.
+ Compressing an input without data (for example `pressio_data_new_empty`) returns an empty output whose size is the maximum compressed size.
+ The compressed data is the bare FSZ stream, identical to the output of `fsz_compress`.
  Decompression needs the `pressio:abs` used for compression and an output with the original type and dimensions.
+ For a value-range relative bound, wrap FSZ in the `pressio` meta-compressor with `pressio:compressor=fsz` and `pressio:rel`.
  The meta-compressor computes the value range on the host, so the input must be in host memory.
+ The plugin keeps an FSZ workspace between calls and recreates it only when the data grows or the current device changes.
  Concurrent calls on one instance take turns on that workspace, while clones receive their own and run independently.

The [LibPressio tutorial](https://github.com/robertu94/libpressio_tutorial) GPU exercise (`exercises/7_gpu_compressors`) has C, C++, and Python versions of this workflow for FSZ.
