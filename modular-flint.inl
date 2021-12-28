/* -*- mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */
// ==========================================================================
// Copyright(c)'2021 by Mahrud Sayrafi
// This file is NOT a part of Givaro, but the majority of it is based
// on modifying givaro/src/kernel/ring/modular-balanced-int64.inl.
// This file is distributed under the GPLv2 license.
// ==========================================================================

#ifndef __modular_flint_INL
#define __modular_flint_INL

#include <cmath> // fmod
#include <flint/fmpz_mat.h>

#define NORMALISE(x)                            \
    {                                           \
        if (x < _mhalfp) x += _p;               \
        else if (x > _halfp) x -= _p;           \
    }

#define NORMALISE_HI(x)                         \
    {                                           \
        if (x > _halfp) x -= _p;                \
    }

// TODO: remove this clause
namespace Givaro {
    int64_t Integer::operator % (unsigned long int l) const
    {
        int64_t res ;
        if (l>0) {
            res = static_cast<int64_t>(this->operator%( static_cast<uint64_t>(l) ) );
        }
        else {
            res = static_cast<int64_t>(this->operator%( static_cast<uint64_t>( -l ) ) );
        }
        return res;
    }
} // namespace Givaro

namespace FFLAS {
    template<>
    inline int Protected::WinogradThreshold (const ARing::ModularFlint<fmpz> & F) { return 1000; } // TODO: figure this out

    template<>
    struct ElementTraits<ARing::ModularFlint<fmpz>> { typedef ElementCategories::ArbitraryPrecIntTag value; };

    // ConvertTo<ElementCategories::ArbitraryPrecIntTag>
    template<>
    struct ModeTraits<ARing::ModularFlint<fmpz>> { typedef ModeCategories::DefaultBoundedTag value; };

    // fgemm for ModularFlint<fmpz> with Winograd Helper: C = alpha * A * B + beta * C
    inline ARing::ModularFlint<fmpz>::Element_ptr fgemm(
        const ARing::ModularFlint<fmpz> &F,
        const FFLAS_TRANSPOSE ta, const FFLAS_TRANSPOSE tb,
        const size_t m, const size_t n, const size_t k,
        ARing::ModularFlint<fmpz>::ConstElement alpha,
        ARing::ModularFlint<fmpz>::ConstElement_ptr Ad, const size_t lda,
        ARing::ModularFlint<fmpz>::ConstElement_ptr Bd, const size_t ldb,
        ARing::ModularFlint<fmpz>::ConstElement beta,
        ARing::ModularFlint<fmpz>::Element_ptr Cd, const size_t ldc,
        MMHelper<ARing::ModularFlint<fmpz>, MMHelperAlgo::Classic, ModeCategories::DefaultTag, ParSeqHelper::Sequential> & H)
    {
        fmpz_mat_t Af, Bf, Cf, Rf;
        fmpz_mat_init(Af, m, k);
        if (ta == FflasNoTrans)
            for (size_t i = 0; i < m; i++)
                Af->rows[i] = (fmpz *)Ad + i * lda;
        else
            for (size_t i = 0; i < m; i++)
            for (size_t j = 0; j < k; j++)
                fmpz_set(&Af->rows[i][j], (fmpz *)Ad + j * lda + i);
        fmpz_mat_init(Bf, k, n);
        if (tb == FflasNoTrans)
            for (size_t i = 0; i < k; i++)
                Bf->rows[i] = (fmpz *)Bd + i * ldb;
        else
            for (size_t i = 0; i < k; i++)
            for (size_t j = 0; j < n; j++)
                fmpz_set(&Bf->rows[i][j], (fmpz *)Bd + j * ldb + i);
        fmpz_mat_init(Cf, m, n);
        for (size_t i = 0; i < m; i++)
            Cf->rows[i] = (fmpz *)Cd + i * ldc;
        // compute the product over Z
        fmpz_mat_init(Rf, m, n);
        fmpz_mat_mul(Rf, Af, Bf);
        // apply alpha and beta
        fmpz* tmp;
        for (size_t i = 0; i < m; i++)
        for (size_t j = 0; j < n; j++)
        {
            tmp = (fmpz *)Cd + i * ldc + j;
            fmpz_fmma(tmp, &alpha, &Rf->rows[i][j], &beta, tmp);
        }
        // MMHelper<RnsDomain, MMHelperAlgo::Classic> H2(Zrns, H.recLevel,H.parseq);
        // fgemm(Zrns,ta,tb,m,n,k,alpha,Ad,lda,Bd,ldb,beta,Cd,ldc,H2);
        // reduce the product mod p
        freduce (F, m, n, Cd, ldc);
        return Cd;
    }

    // fgemm for ModularFlint<fmpz>: Y = alpha * A * X + beta * Y
    inline ARing::ModularFlint<fmpz>::Element_ptr fgemv(
        const ARing::ModularFlint<fmpz> &F,
        const FFLAS_TRANSPOSE ta,
        const size_t m, const size_t n,
        ARing::ModularFlint<fmpz>::ConstElement alpha,
        ARing::ModularFlint<fmpz>::ConstElement_ptr A, const size_t lda,
        ARing::ModularFlint<fmpz>::ConstElement_ptr X, const size_t ldx,
        ARing::ModularFlint<fmpz>::ConstElement beta,
        ARing::ModularFlint<fmpz>::Element_ptr Y, const size_t ldy,
        MMHelper<ARing::ModularFlint<fmpz>, MMHelperAlgo::Classic> &H)
    {
        MMHelper<ARing::ModularFlint<fmpz>, MMHelperAlgo::Winograd> H2(H);
        fgemm(F,ta,FflasNoTrans,(ta==FflasNoTrans)?m:n,1,(ta==FflasNoTrans)?n:m,alpha,A,lda,X,ldx,beta,Y,ldy,H2);
        return Y;
    }

    inline size_t bitsize(
        const ARing::ModularFlint<fmpz>& F,
        const size_t M, const size_t N,
        ARing::ModularFlint<fmpz>::ConstElement_ptr A, size_t lda) {
        fmpz max = FLINT_MAX(F.maxElement(), FLINT_ABS(F.minElement()));
        return fmpz_bits(&max);
    }
} // namespace FFLAS

namespace ARing
{

    //----- Classic arithmetic

    inline ModularFlint<fmpz>::Element&
    ModularFlint<fmpz>::mul(Element& r, const Element& a, const Element& b) const
    {
        fmpz_mod_mul(&r, &a, &b, _ctx);
        // Element q = static_cast<Element>(double(a) * double(b) * _dinvp);
        // r = static_cast<Element>(a * b - q * _p);
        // NORMALISE(r);
        return r;
    }

    inline ModularFlint<fmpz>::Element&
    ModularFlint<fmpz>::div(Element& r, const Element& a, const Element& b) const
    {
        Element tmp;
        return mul (r, a, inv(tmp, b));
    }

    inline ModularFlint<fmpz>::Element&
    ModularFlint<fmpz>::add(Element& r, const Element& a, const Element& b) const
    {
        fmpz_mod_add(&r, &a, &b, _ctx);
        // r = a + b;
        // NORMALISE(r);
        return r;
    }

    inline ModularFlint<fmpz>::Element&
    ModularFlint<fmpz>::sub(Element& r, const Element& a, const Element& b) const
    {
        fmpz_mod_sub(&r, &a, &b, _ctx);
        // r = a - b;
        // NORMALISE(r);
        return r;
    }

    inline ModularFlint<fmpz>::Element&
    ModularFlint<fmpz>::neg(Element& r, const Element& a) const
    {
        fmpz_mod_neg(&r, &a, _ctx);
        return r; // = -a;
    }

    inline ModularFlint<fmpz>::Element&
    ModularFlint<fmpz>::inv(Element& r, const Element& a) const
    {
        fmpz_mod_inv(&r, &a, _ctx);
        // NORMALISE(r);
        return r;
    }

    inline bool ModularFlint<fmpz>::isUnit(const Element& a) const
    {
        fmpz_t d;
        fmpz_init(d);
        fmpz_gcd(d, &a, &_p);
        return fmpz_is_pm1(d);
    }

    inline ModularFlint<fmpz>::Element&
    ModularFlint<fmpz>::mulin(Element& r, const Element& a) const
    {
        return mul(r, r, a);
    }

    inline ModularFlint<fmpz>::Element&
    ModularFlint<fmpz>::divin(Element& r, const Element& a) const
    {
        return div(r, r, a);
    }

    inline ModularFlint<fmpz>::Element&
    ModularFlint<fmpz>::addin(Element& r, const Element& a) const
    {
        return add(r, r, a);
    }

    inline ModularFlint<fmpz>::Element&
    ModularFlint<fmpz>::subin(Element& r, const Element& a) const
    {
        return sub(r, r, a);
    }

    inline ModularFlint<fmpz>::Element&
    ModularFlint<fmpz>::negin(Element& r) const
    {
        return neg(r, r);
    }

    inline ModularFlint<fmpz>::Element&
    ModularFlint<fmpz>::invin(Element& r) const
    {
        return inv(r, r);
    }

    //----- Special arithmetic

    inline ModularFlint<fmpz>::Element&
    ModularFlint<fmpz>::axpy(Element& r, const Element& a, const Element& x, const Element& y) const
    {
        // q could be off by (+/-) 1
        Element q = static_cast<Element>(((((double) a) * ((double) x)) + (double) y) * _dinvp);
        r = static_cast<Element>(a * x + y - q * _p);
        NORMALISE(r);
        return r;
    }

    inline ModularFlint<fmpz>::Element&
    ModularFlint<fmpz>::axpyin(Element& r, const Element& a, const Element& x) const
    {
        // q could be off by (+/-) 1
        Element q = static_cast<Element>(((((double) a) * ((double) x)) + (double) r) * _dinvp);
        r = static_cast<Element>(a * x + r - q * _p);
        NORMALISE(r);
        return r;
    }

    inline ModularFlint<fmpz>::Element&
    ModularFlint<fmpz>::axmy(Element& r, const Element& a, const Element& x, const Element& y) const
    {
        // q could be off by (+/-) 1
        Element q = static_cast<Element>(((((double) a) * ((double) x)) - (double) y) * _dinvp);
        r = static_cast<Element>(a * x - y - q * _p);
        NORMALISE(r);
        return r;
    }

    inline ModularFlint<fmpz>::Element&
    ModularFlint<fmpz>::axmyin(Element& r, const Element& a, const Element& x) const
    {
        // q could be off by (+/-) 1
        Element q = static_cast<Element>(((((double) a) * ((double) x)) - (double) r) * _dinvp);
        r = static_cast<Element>(a * x - r - q * _p);
        NORMALISE(r);
        return r;
    }

    inline ModularFlint<fmpz>::Element&
    ModularFlint<fmpz>::maxpy(Element& r, const Element& a, const Element& x, const Element& y) const
    {
        return negin(axmy(r, a, x, y));
    }

    inline ModularFlint<fmpz>::Element&
    ModularFlint<fmpz>:: maxpyin(Element& r, const Element& a, const Element& x) const
    {
        return negin(axmyin(r, a, x));
    }

    //----- Init

    inline ModularFlint<fmpz>::Element&
    ModularFlint<fmpz>::init(Element& x) const
    {
        return x = 0;
    }

    inline ModularFlint<fmpz>::Element&
    ModularFlint<fmpz>::init(Element& x, const float y) const
    {
        x = static_cast<Element>(fmod(y, double(_p)));
        NORMALISE(x);
        return x;
    }

    inline ModularFlint<fmpz>::Element&
    ModularFlint<fmpz>::init(Element& x, const double y) const
    {
        fmpz_init(&x);
        fmpz_set_d(&x, y);
        fmpz_mod(&x, &x, &_p);
        // x = static_cast<Element>(fmod(y, double(_p)));
        // NORMALISE(x);
        return x;
    }

    inline ModularFlint<fmpz>::Element&
    ModularFlint<fmpz>::init(Element& x, const int64_t y) const
    {
        fmpz_init_set_si(&x, y);
        fmpz_mod(&x, &x, &_p);
        // x = static_cast<Element>(y % _p);
        // NORMALISE_HI(x);
        return x;
    }

    inline ModularFlint<fmpz>::Element&
    ModularFlint<fmpz>::init(Element& x, const uint64_t y) const
    {
        fmpz_init_set_ui(&x, y);
        fmpz_mod(&x, &x, &_p);
        // x = static_cast<Element>(y % _p);
        // NORMALISE_HI(x);
        return x;
    }

    inline ModularFlint<fmpz>::Element&
    ModularFlint<fmpz>::assign(Element& x, const Element& y) const
    {
        fmpz_init_set(&x, &y);
        // fmpz_mod(&x, &x, &_p);
        return x;
    }

    //----- Reduce

    inline ModularFlint<fmpz>::Element&
    ModularFlint<fmpz>::reduce(Element& x, const Element& y) const
    {
        fmpz_mod(&x, &y, &_p);
        // x = y % _p;
        // NORMALISE(x);
        return x;
    }

    inline ModularFlint<fmpz>::Element&
    ModularFlint<fmpz>::reduce(Element& x) const
    {
        fmpz_mod(&x, &x, &_p);
        // x %= _p;
        // NORMALISE(x);
        return x;
    }

    //----- IO

    inline std::ostream&
    ModularFlint<fmpz>::write(std::ostream& os) const
    {
        return os << "ModularFlint<fmpz> modulo " << _p;
    }

    inline std::ostream&
    ModularFlint<fmpz>::write(std::ostream& os, const Element& x) const
    {
        return os << x;
    }

    inline std::istream&
    ModularFlint<fmpz>::read(std::istream& is, Element& x) const
    {
        Element tmp;
        is >> tmp;
        init(x, tmp);
        return is;
    }

} // namespace ARing

#undef NORMALISE
#undef NORMALISE_HI

#endif // __modular_flint_INL
// vim:sts=4:sw=4:ts=4:et:sr:cino=>s,f0,{0,g0,(0,\:0,t0,+0,=s
