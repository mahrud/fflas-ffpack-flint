# FFLAS-FFPACK 🤝 FLINT

`test.cpp` contains a proof of concept that the templated C++ library [FFLAS-FFPACK](https://github.com/linbox-team/fflas-ffpack) can be used to perform fast linear algebra computations (at least rank!) using [FLINT](https://flintlib.org/)'s `fmpz` rather than [Givaro](https://github.com/linbox-team/givaro)'s 'Integer`.

## Building

```
$ make all
clang++ test.cpp -o test -I`brew --prefix`/include -I/usr/include -L`brew --prefix`/lib -L/usr/lib64 -lflint -lgmp -lopenblas -g
```

## Checking

A successful compilation of a program written using a templated library is in some sense a proof of success. But for sanity, here is a check for computing rank of an 11x11 matrix over ZZ/32003:
```
$ make check
./test 32003 mat11.sms
A = 
     0      0      2      3      0      0      0      0      0      1      0
     0      0      0      0      0      0      0      0      0      0      0
     2      0    888      1     -1      0      0      0      0      0      6
     3      0      1      4      0      0     12      0      0    -13      0
     0      0     -1      0      0      0      0      0      0      0      0
     0      0      0      0      0      1      0      1      0      1      0
     0      0      0     12      0      0      0      0      0      0      0
     0      0      0      0      0      1      0    500    400    300    200
     0      0      0      0      0      0      0    400      0      0      0
     1      0      0    -13      0      1      0    300      0     10      1
     0      0      6      0      0      0      0    200      0      1      0

rank A = 9
```

## Benchmarks

See [here](bench).

## Debugging

1. Run the preprocessor
```
$ make pp
clang++ test.cpp -o test.cc -I`brew --prefix`/include -I/usr/include -g -E
```
2. Run the compiler
```
$ make cc
clang++ test.cc -o test.o -Wno-parentheses-equality --save-temps -gdwarf -g -c
```
3. Run the linker
```
$ make ll
clang++ test.o -o test -L`brew --prefix`/lib -L/usr/lib64 -lflint -lgmp -lopenblas
```

## TODO

- [ ] There are a number of segmentation faults and double free bugs revealed by the benchmarks.
- [ ] Extend `ModularFlint` to other FLINT types, in particular extension fields.
- [ ] A limited number of inlined functions involving Givaro's `Integer` type seem to be involved in the object file. Investigate and remove them.
- [x] Benchmark linear algebra computations using FLINT's `fmpz` vs. Givaro's `Integer`.
