#include <list>
#include <functional>
#include "../inc/PoissonEqn.h"

extern int NumOfPnt, NumOfFace, NumOfCell;
extern NaturalArray<Point> pnt;
extern NaturalArray<Face> face;
extern NaturalArray<Cell> cell;
extern NaturalArray<Patch> patch;

static const size_t ref_cell = 0; /// 0-based
static const Scalar ref_val = 0.0; /// Pa

static bool is_ref_row(const Eigen::Triplet<Scalar> &x)
{
    return x.row() == ref_cell;
}

/// Borrow from SciPy V1.4.1
/// https://github.com/scipy/scipy/blob/v1.4.1/scipy/sparse/sparsetools/coo.h
template <class I, class T>
static void coo2csr(const I n_row, const I n_col, const I nnz, const I *Ai, const I *Aj, const T *Ax, I *Bp, I *Bj, T *Bx)
{
    //compute number of non-zero entries per row of A
    std::fill(Bp, Bp + n_row, 0);

    for (I n = 0; n < nnz; n++) {
        Bp[Ai[n]]++;
    }

    //cumsum the nnz per row to get Bp[]
    for (I i = 0, cumsum = 0; i < n_row; i++) {
        I temp = Bp[i];
        Bp[i] = cumsum;
        cumsum += temp;
    }
    Bp[n_row] = nnz;

    //write Aj,Ax into Bj,Bx
    for (I n = 0; n < nnz; n++) {
        I row = Ai[n];
        I dest = Bp[row];

        Bj[dest] = Aj[n];
        Bx[dest] = Ax[n];

        Bp[row]++;
    }

    for (I i = 0, last = 0; i <= n_row; i++) {
        I temp = Bp[i];
        Bp[i] = last;
        last = temp;
    }

    //now Bp,Bj,Bx form a CSR representation (with possible duplicates)
}

void PC_calcSteadyPart(SX_MAT &B)
{
    /// Calculate raw triplets
    std::list<Eigen::Triplet<Scalar>> coefficient; /// (I, J, val), I and J are 0-based.

    /// Calculate original coefficients for each cell.
    for (const auto &C : cell)
    {
        /// Initialize coefficient baseline.
        std::map<int, Scalar> cur_coefficient;
        cur_coefficient[C.index] = 0.0;
        for (auto F : C.adjCell)
        {
            if (F)
                cur_coefficient[F->index] = 0.0;
        }

        /// Compute coefficient contributions.
        const size_t N_C = C.surface.size();
        for (size_t f = 0; f < N_C; ++f)
        {
            const auto &E_f = C.Se.at(f);

            auto curFace = C.surface.at(f);
            if (curFace->at_boundary) /// Boundary Case.
            {
                /// When p is given on boundary, p' is 0-value there;
                /// When p is not determined on boundary, p' is 0-gradient there,
                /// thus contribution is 0 and no need to handle.
                if (curFace->parent->p_prime_BC == Dirichlet)
                    cur_coefficient[C.index] += E_f.norm() / (curFace->centroid - C.centroid).norm();
            }
            else /// Internal Case.
            {
                auto F = C.adjCell.at(f);
                if (!F)
                    throw empty_connectivity(C.index);

                const Scalar mul = E_f.norm() / (F->centroid - C.centroid).norm();
                cur_coefficient[C.index] += mul;
                cur_coefficient[F->index] -= mul;
            }
        }

        /// Record current line.
        /// Convert index to 0-based.
        for (const auto &it : cur_coefficient)
            coefficient.emplace_back(C.index - 1, it.first - 1, it.second);
    }

    /// Allocate storage
    B = sx_mat_struct_create(NumOfCell, NumOfCell, coefficient.size());

    /// Transform
    auto Ai = new SX_INT[B.num_nnzs];
    auto Aj = new SX_INT[B.num_nnzs];
    auto Ax = new SX_FLT[B.num_nnzs];

    int n = 0;
    for (const auto &e : coefficient)
    {
        Ai[n] = e.row();
        Aj[n] = e.col();
        Ax[n] = e.value();
        ++n;
    }

    /// COO to CSR
    coo2csr<SX_INT, SX_FLT>(B.num_rows, B.num_cols, B.num_nnzs, Ai, Aj, Ax, B.Ap, B.Aj, B.Ax);

    /// Release
    delete[] Ai;
    delete[] Aj;
    delete[] Ax;
}

void PC_updateDiagonalPart(SX_MAT *A, SX_VEC *base, SX_VEC *variation)
{
    for (size_t i = 0; i < NumOfCell; ++i)
    {
        const size_t ibegin = A->Ap[i];
        const size_t iend = A->Ap[i + 1];
        for (size_t k = ibegin; k < iend; ++k)
        {
            const size_t j = A->Aj[k];
            if ((j - i) == 0)
            {
                A->Ax[k] = sx_vec_get_entry(base, i) + sx_vec_get_entry(variation, i);
                break;
            }
        }
    }
}

void PC_updateRHS(SX_VEC *b, Scalar dt)
{
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> rhs;
    rhs.resize(NumOfCell, Eigen::NoChange);

    rhs.setZero();
    for (const auto &C : cell)
    {
        const auto N_C = C.surface.size();
        const auto lci = C.index - 1; /// 0-based
        auto &cur_rhs = rhs(lci);

        /// Diffusion part
        cur_rhs = -C.dmdt / dt;

        /// Convection part
        for (size_t f = 0; f < N_C; ++f)
        {
            const auto &S_f = C.S.at(f);
            auto curFace = C.surface.at(f);
            cur_rhs -= curFace->drhodp_prev * curFace->p_prime * curFace->U_star.dot(S_f) / dt;
        }

        /// Additional contribution due to cross-diffusion
        for (size_t f = 0; f < N_C; ++f)
        {
            const auto &T_f = C.St.at(f);
            auto curFace = C.surface.at(f);
            if(curFace->at_boundary)
            {
                if(curFace->parent->p_prime_BC == Dirichlet)
                    cur_rhs += curFace->grad_p_prime.dot(T_f);
            }
            else
                cur_rhs += curFace->grad_p_prime.dot(T_f);
        }
    }

    for (SX_INT i = 0; i < NumOfCell; ++i)
        sx_vec_set_entry(b, i, rhs(i));
}

void PC_prepareSolver(SX_MAT &A, SX_AMG &mg)
{
    SX_AMG_PARS pars;

    sx_amg_pars_init(&pars);
    pars.maxit = 200;
    pars.verb = 0;
    pars.cs_type = SX_COARSE_RS;
    pars.interp_type = SX_INTERP_STD;
    pars.strong_threshold = 0.3;
    pars.trunc_threshold = 0.2;

    sx_amg_pars_print(&pars);

    sx_amg_setup(&mg, &A, &pars);
}
