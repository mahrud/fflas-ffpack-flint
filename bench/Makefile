N = 1000
bench-opts = -q 32003 -m $(N) -n $(N) -r $(N) -i 1
build-opts = -g -I`brew --prefix`/include -I/usr/include -L`brew --prefix`/lib -L/usr/lib64 -lflint -lgmp -lopenblas -lgivaro -fopenmp -lomp
lib-opts = LD_LIBRARY_PATH=`brew --prefix`/lib

all: bench-givaro bench-flint

# Building executables
bench-givaro: rank.cpp
	clang++ rank.cpp -o bench-givaro $(build-opts)

bench-flint bench-flint-fflas: rank.cpp
	clang++ rank.cpp -o bench-flint $(build-opts) -DUSE_FLINT_RANK

# Individual benchmarks
# Time: 0.472294 Gfops: 11.2924 (with ModularBalanced<double>)
# Time: 2.67511 Gfops: 1.99368 (with ModularBalanced<int64_t>)
fflas-givaro: bench-givaro
	$(lib-opts) ./bench-givaro $(bench-opts)

# Time: 2.77078 Gfops: 1.92485
fflas-flint: bench-flint
	$(lib-opts) ./bench-flint $(bench-opts)

# Time: 5.26112 Gfops: 1.01373
flint: bench-flint
	$(lib-opts) ./bench-flint $(bench-opts) -f Y

# Ranged benchmarks
benchmark-%:
	for n in {10..100..10}; do \
#	for n in {500..5000..500}; do \
#	for n in {100..1000..100}; do \
		echo -n "R = $$n ... "; \
		export T="`make $* -e N=$$n | grep Time | cut -d' ' -f4`"; \
		echo "$$T Gfops"; echo "$$n $$T" >> $@.dat; \
	done

# Plotting targets
plot-%: plot-%.dat
	gnuplot -p -e "plot '$<' smooth bezier title '$*'"

plot:
	gnuplot -p -e " \
		set title 'Benchmarking matrix rank implementations'; \
		set xlabel 'Rank of the matrix'; \
		set ylabel 'Gfops'; \
		plot for [filename in system('ls *.dat | xargs')] filename smooth bezier title filename"

clean:
	rm bench-* plot-*
