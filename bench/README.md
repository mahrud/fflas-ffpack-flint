### Benchmarks

The plot below compares the speed of computing rank of a full rank NxN matrix using three different algorithms:
- FLINT's `fq_nmod_mat_rank` using FLINT's `fmpz`
- `FFPACK::PLUQ` using FLINT's `fmpz`
- `FFPACK::PLUQ` using Givaro's `Integer`

by comparing the number of field operations per second (so higher is better).

![](benchmark.svg)
