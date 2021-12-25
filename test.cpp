// ==========================================================================
// Copyright(c)'2021 by Mahrud Sayrafi
// This file is NOT a part of FFLAS-FFPACK, but the majority of it is based
// on modifying fflas_ffpack/examples/rank.C.
// This file is distributed under the GPLv2 license.
// ==========================================================================

#include <fflas-ffpack/fflas-ffpack.h>
#include <flint/flint.h>
#include <flint/fmpz.h>
#include <flint/fq_nmod.h>
#include <iostream>

#include "modular-flint.h"

/**
 * This example computes the rank of a matrix
 * over a defined finite field implemented by Flint.
 *
 * Outputs the rank.
 */
int main(int argc, char** argv)
{
    if (argc != 3) {
        std::cerr << "Usage: rank <p> <matrix>" << std::endl;
        return -1;
    }

    int p0 = atoi(argv[1]);
    std::string file = argv[2];

    // Creating the finite field Z/pZ
    fmpz p;
    fmpz_init_set_ui(&p, p0);
    ARing::ModularFlint<fmpz> F(p);

    // Reading the matrix from a file
    fmpz* A;
    size_t m, n;
    FFLAS::ReadMatrix(file.c_str(), F, m, n, A);
    FFLAS::WriteMatrix(std::cout << "A = " << std::endl, F, m, n, A, m, FFLAS::FflasAuto)<<std::endl;

    // Compute and print the rank
    size_t r = FFPACK::Rank(F, m, n, A, n);
    std::cout << "rank A = " << r << std::endl;

    // Clear the memory
    FFLAS::fflas_delete(A);

    return 0;
}
