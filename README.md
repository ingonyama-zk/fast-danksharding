# Fast Danksharding

This implementation is meant to run entirely on an NVIDIA GPU.The code is a highly parallelized implementation of the diagram below, and heavily rely on  [ICICLE](https://github.com/ingonyama-zk/icicle). For implementation details please refer to our [companion blog post](https://medium.com/@ingonyama/fast-danksharding-using-icicle-6411565bb245) 

 <div align="center">
<img width="618" alt="Screenshot 2023-03-08 at 15 34 50" src="https://user-images.githubusercontent.com/2446179/223727252-6e94d18f-0653-4c0d-87ad-5c7d82c0ea54.png">
</div>

## Build and usage

> NOTE: [CUDA 12+], [NVCC], [Rust], [C++17] are prerequisites for building. Python 3.10 and up with [numpy](https://numpy.org/install/) and [tqdm](https://tqdm.github.io/) packages for test data generation. Tested on Ubuntu 20.04+

1. ICICLE library supports BLS12-381 as default curve, that is the curve in use for danksharding
2. Go into git dir init the submodule and update
3. Generate test vectors
4. Build and run fast-danksharding using cargo

```sh
git submodule init
git submodule update
cd kzg_data_availability
python3 tests.py; cd ..
cargo run --release
```

You'll find a release ready binary at `target/release/`.

## Contributing

Join our [Discord Server](https://discord.gg/Y4SkbDf2Ff) and find us on the icicle channel. We will be happy to work together to support your use case and talk features, bugs and design.

## License

ICICLE is distributed under the terms of the MIT License.

See [LICENSE-MIT][LMIT] for details.

<!-- Begin Links -->
[BLS12-381]: https://github.com/ingonyama-zk/icicle/blob/main/icicle/curves/bls12_381.cuh
[CUDA 12+]: https://developer.nvidia.com/cuda-downloads
[NVCC]: https://docs.nvidia.com/cuda/#installation-guides
[Rust]: https://www.rust-lang.org/
[C++17]: https://en.cppreference.com/w/cpp/17
[CRV_TEMPLATE]: ./icicle/curves/curve_template.cuh
[CRV_CONFIG]: ./icicle/curves/curve_config.cuh
[B_SCRIPT]: ./build.rs
[FDI]: https://github.com/ingonyama-zk/fast-danksharding
[CONT]: ./CONTRIBUTING.md
[LMIT]: ./LICENSE
<!-- End Links -->
