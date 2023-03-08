# Fast Danksharding

## Ingonyama danksharding scheme implementation
Using ICICLE, we have implemented a flow that can run entirely in the GPU, with only the relevant outputs to be sent to a host machine and populated on the Ethereum network.
Our work focuses on the smallest functions and building blocks that are relevant to the Danksharding builder. The CUDA functions that we designed, along with bindings to our Rust wrappers are capable of executing this process.

 <div align="center">
<img width="618" alt="Screenshot 2023-03-08 at 15 34 50" src="https://user-images.githubusercontent.com/2446179/223727252-6e94d18f-0653-4c0d-87ad-5c7d82c0ea54.png">
</div>

## Build and usage

> NOTE: [NVCC] and [Rust] are prerequisites for building.

1. ICICLE library supports BLS12-381 as default curve, that is the curve in use for danksharding
2. To build and run fast-danksharding using cargo

```sh
cargo run --release
```

You'll find a release ready binary at `target/release/`.

## Contributing

If you would like to contribute with code, check the [CONTRIBUTING.md][CONT] file for further info about the development environment.

## License

ICICLE is distributed under the terms of the MIT License.

See [LICENSE-MIT][LMIT] for details.

<!-- Begin Links -->
[BLS12-381]: https://github.com/ingonyama-zk/icicle/blob/main/icicle/curves/bls12_381.cuh
[NVCC]: https://docs.nvidia.com/cuda/#installation-guides
[Rust]: https://www.rust-lang.org/
[CRV_TEMPLATE]: ./icicle/curves/curve_template.cuh
[CRV_CONFIG]: ./icicle/curves/curve_config.cuh
[B_SCRIPT]: ./build.rs
[FDI]: https://github.com/ingonyama-zk/fast-danksharding
[CONT]: ./CONTRIBUTING.md
[LMIT]: ./LICENSE
<!-- End Links -->
