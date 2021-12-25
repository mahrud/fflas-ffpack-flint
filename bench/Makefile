bench-givaro: benchmark-pluq.C
	clang++ benchmark-pluq.C -o bench-givaro -I`brew --prefix`/include -I/usr/include -L`brew --prefix`/lib -L/usr/lib64 -lflint -lgmp -lopenblas -lgivaro -g

bench-flint: benchmark-pluq.C
	clang++ benchmark-pluq.C -o bench-flint -I`brew --prefix`/include -I/usr/include -L`brew --prefix`/lib -L/usr/lib64 -lflint -lgmp -lopenblas -lgivaro -g -DUSE_FLINT_RANK

# Time: 0.472294 Gfops: 11.2924 (with ModularBalanced<double>)
# Time: 2.67511 Gfops: 1.99368 (with ModularBalanced<int64_t>)
givaro:
	LD_LIBRARY_PATH=`brew --prefix`/lib ./bench-givaro

# Time: 5.26112 Gfops: 1.01373
flint:
	LD_LIBRARY_PATH=`brew --prefix`/lib ./bench-flint -f Y

# Time: 2.77078 Gfops: 1.92485
flint-fflas:
	LD_LIBRARY_PATH=`brew --prefix`/lib ./bench-flint
