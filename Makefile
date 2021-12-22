all:
	clang++ test.cpp -o test -I`brew --prefix`/include -I/usr/include -L`brew --prefix`/lib -L/usr/lib64 -lflint -lgmp -lopenblas -g

# preprocess only
pp:
	clang++ test.cpp -o test.cc -I`brew --prefix`/include -I/usr/include -g -E

# compile only
cc:
	clang++ test.cc -o test.o -Wno-parentheses-equality --save-temps -gdwarf -g -c

# link only
ll:
	clang++ test.o -o test -L`brew --prefix`/lib -L/usr/lib64 -lflint -lgmp -lopenblas

check:
	LD_LIBRARY_PATH=`brew --prefix`/lib ./test 32003 mat11.sms

iwyu:
	include-what-you-use -Xiwyu --max_line_length=120 test.cpp -o test -I`brew --prefix`/include -I/usr/include
