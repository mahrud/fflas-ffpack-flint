/* -*- mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */
// ==========================================================================
// This file is based on fflas_ffpack/benchmarks/benchmark-pluq.C.
// Modified and redistributed under LGPLv2.1+
// ==========================================================================

//#define __FFLASFFPACK_USE_OPENMP
//#define __FFLASFFPACK_USE_TBB

// declare that the call to openblas_set_numthread will be made here,
// hence don't do it everywhere in the call stack
#define __FFLASFFPACK_OPENBLAS_NT_ALREADY_SET 1

#include "fflas-ffpack/fflas-ffpack-config.h"
#include <givaro/gfq.h>
#include <givaro/modular.h>
#include <givaro/modular-integer.h>
#include <givaro/givranditer.h>

#include "fflas-ffpack/config-blas.h"
#include "fflas-ffpack/fflas/fflas.h"
#include "fflas-ffpack/utils/timer.h"
#include "fflas-ffpack/utils/fflas_randommatrix.h"
#include "fflas-ffpack/utils/args-parser.h"
#include "fflas-ffpack/ffpack/ffpack.h"

#include <flint/flint.h>
#include <flint/fmpz.h>
#include <flint/fmpz_mat.h>
#include <flint/fq_nmod.h>
#include <flint/fq_nmod_mat.h>

#include "../modular-flint.h"

#include <iostream>

using namespace std;

#ifdef USE_FLINT_RANK
typedef ARing::ModularFlint<fmpz> Field;
#else
//typedef Givaro::Modular<Givaro::Integer> Field; // crashes
//typedef Givaro::ModularBalanced<double> Field;
typedef Givaro::ModularBalanced<int64_t> Field;
//typedef Givaro::ModularBalanced<float> Field;
//typedef Givaro::GFqDom<int32_t> Field; // doesn't compile
//typedef Givaro::ZRing<double> Field;
//typedef Givaro::UnparametricZRing<double> Field;
#endif

void Rec_Initialize(Field &F, Field::Element * C, size_t m, size_t n, size_t ldc)
{
    if(std::min(m,n) <= ldc/NUM_THREADS){
        for(size_t i=0; i<m; i++)
            FFLAS::fzero(F, 1, n, C+i*n, n);
    }
    else{
        size_t M2 = m >> 1;
        size_t N2 = n >> 1;
        typename Field::Element * C2 = C + N2;
        typename Field::Element * C3 = C + M2*ldc;
        typename Field::Element * C4 = C3 + N2;

        SYNCH_GROUP(
            TASK(MODE(CONSTREFERENCE(F)), Rec_Initialize(F,C,M2,N2, ldc););
            TASK(MODE(CONSTREFERENCE(F)), Rec_Initialize(F,C2,M2,n-N2, ldc););
            TASK(MODE(CONSTREFERENCE(F)), Rec_Initialize(F,C3,m-M2,N2, ldc););
            TASK(MODE(CONSTREFERENCE(F)), Rec_Initialize(F,C4,m-M2,n-N2, ldc););
            );
    }
}

int main(int argc, char** argv) {

#ifdef __FFLASFFPACK_OPENBLAS_NUM_THREADS
    openblas_set_num_threads(__FFLASFFPACK_OPENBLAS_NUM_THREADS);
#endif

    size_t iter = 3 ;
    bool flint = false;
    int q = 131071 ;
    int m = 2000 ;
    int n = 2000 ;
    int r = 2000 ;
    int v = 0;
    Argument as[] = {
        { 'f', "-f F", "Use Flint for computing rank.", TYPE_BOOL , &flint },
        { 'q', "-q Q", "Set the field characteristic (-1 for random).",         TYPE_INT , &q },
        { 'm', "-m M", "Set the row dimension of A.",      TYPE_INT , &m },
        { 'n', "-n N", "Set the col dimension of A.",      TYPE_INT , &n },
        { 'r', "-r R", "Set the rank of matrix A.",            TYPE_INT , &r },
        { 'i', "-i I", "Set number of repetitions.",            TYPE_INT , &iter },
        { 'v', "-v V", "Set 1 if need verification of result else 0.",            TYPE_INT , &v },
        END_OF_ARGUMENTS
    };
    FFLAS::parseArguments(argc,argv,as);
    if (r > std::min(m, n)) r = std::min(m, n);

    // FIXME: why does this ometimes segfault?
    std::cout << "Defining F(q) ... ";
    Field F(q);
    std::cout << "done." << std::endl;

    size_t R;
    Field::Element_ptr A, Acop;
    A = FFLAS::fflas_new(F,m,n);

    fmpz_t q0;
    fmpz_set_ui(q0, q);
    fq_nmod_ctx_t ctx;
    fq_nmod_ctx_init(ctx, q0, 1, "a");
    fq_nmod_mat_t Amat, Amatcop;
    fq_nmod_mat_init(Amatcop, m, n, ctx);

    PAR_BLOCK{
        Rec_Initialize(F, A, m, n, n);
        FFPACK::RandomMatrixWithRankandRandomRPM (F, m, n, r, A, n);
    }

    FFLAS::Timer chrono;
    double *time = new double[iter];

    enum FFLAS::FFLAS_DIAG diag = FFLAS::FflasNonUnit;
    size_t maxP, maxQ;
    maxP = m;
    maxQ = n;

    size_t *P = FFLAS::fflas_new<size_t>(maxP);
    size_t *Q = FFLAS::fflas_new<size_t>(maxQ);

    if (flint) {
#ifdef USE_FLINT_RANK
	    for(int i=0;i<m;++i)
        for(int j=0;j<n;++j)
        {
            fq_nmod_struct* a = fq_nmod_mat_entry(Amatcop, i, j);
            fq_nmod_set_fmpz(a, A + i*n + j, ctx);
        }
	    fq_nmod_mat_init_set(Amat, Amatcop, ctx);
#if 0
	    FFLAS::WriteMatrix(std::cout << "A = " << std::endl, F, m, n, A, m, FFLAS::FflasAuto) << std::endl;
	    fq_nmod_mat_print_pretty(Amatcop, ctx);
	    std::cout << std::endl;
#endif
#endif
    } else {
        Acop = FFLAS::fflas_new(F,m,n);
        FFLAS::ParSeqHelper::Parallel<FFLAS::CuttingStrategy::Recursive,
                                      FFLAS::StrategyParameter::Threads> parH;
        PARFOR1D(i,(size_t)m,parH,
                 FFLAS::fassign(F, n, A + i*n, 1, Acop + i*n, 1);
                 // for (size_t j=0; j<(size_t)n; ++j)
                 //     Acop[i*n+j]= A[i*n+j];
            );
    }

    for (size_t i=0;i<=iter;++i)
    {
        if (flint) fq_nmod_mat_set(Amat, Amatcop, ctx);
        else {
        PARFOR1D(j,maxP,parH, P[j]=0; );
        PARFOR1D(j,maxQ,parH, Q[j]=0; );
        PARFOR1D(k,(size_t)m,parH,
                 FFLAS::fassign(F, n, Acop + k*n, 1, A + k*n, 1);
                 // for (size_t j=0; j<(size_t)n; ++j)
                 //     F.assign( A[k*n+j] , Acop[k*n+j]) ;
                );
        }

        chrono.clear();

        if (i) chrono.start();
        if (flint)
            R = fq_nmod_mat_rank(Amat, ctx);
        else
            R = FFPACK::PLUQ(F, diag, m, n, A, n, P, Q);
//          R = FFPACK::pPLUQ(F, diag, m, n, A, n, P, Q);
//          R = FFPACK::LUdivine(F, diag, FFLAS::FflasNoTrans, m, n, A, n, P, Q);
        if (i) {chrono.stop(); time[i-1]=chrono.realtime();}

//        std::cout << "rank = " << R << std::endl;
        assert(R == r);
    }

    std::sort(time, time+iter);
    double mediantime = time[iter/2];
    delete[] time;
    // -----------
    // Standard output for benchmark - Alexis Breust 2014/11/14
#define CUBE(x) ((x)*(x)*(x))
    double gflop =  2.0/3.0*CUBE(double(r)/1000.0) +2*m/1000.0*n/1000.0*double(r)/1000.0  - double(r)/1000.0*double(r)/1000.0*(m+n)/1000;
    std::cout << "Time: " << mediantime
    << " Gfops: " << gflop / mediantime;
    FFLAS::writeCommandString(std::cout, as) << std::endl;

    FFLAS::fflas_delete (P);
    FFLAS::fflas_delete (Q);
    FFLAS::fflas_delete (A);
    FFLAS::fflas_delete (Acop);

    return 0;
}
// vim:sts=4:sw=4:ts=4:et:sr:cino=>s,f0,{0,g0,(0,\:0,t0,+0,=s
