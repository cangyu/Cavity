#include <iostream>
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

static void gen_coef_triplets
(
    std::list<Eigen::Triplet<Scalar>> &coef,
    const std::vector<Scalar> &ud,
    const std::vector<Vector> &cvec
)
{
    /// Calculate original coefficients for each cell.
    for (const auto &C : cell)
    {
        /// Contribution from unsteady term.
        std::map<int, Scalar> cur_coef;
        cur_coef[C.index] = ud.at(C.index-1);
        for (auto F : C.adjCell)
        {
            if (F)
                cur_coef[F->index] = 0.0;
        }

        /// Contribution from diffusion term.
        const auto N_C = C.surface.size();
        for (int f = 1; f <= N_C; ++f)
        {
            auto curFace = C.surface(f);
            const auto &E_f = C.Se(f);
            const auto &d_f = C.d(f);

            if (curFace->at_boundary) /// Boundary Case.
            {
                switch (curFace->parent->p_BC)
                {
                case Dirichlet:
                    cur_coef[C.index] +=E_f.norm() / (curFace->centroid - C.centroid).norm() ; /// When p is given on boundary, p' is 0-value there.
                    break;
                case Neumann:
                    break;/// When p is not determined on boundary, p' is 0-gradient there. Thus, no contribution.
                default:
                    throw unsupported_boundary_condition(curFace->parent->p_BC);
                }
            }
            else /// Internal Case.
            {
                auto F = C.adjCell(f);
                if (!F)
                    throw inconsistent_connectivity("Cell shouldn't be empty!");

                cur_coef[C.index] += E_f.norm() / (F->centroid - C.centroid).norm();
                cur_coef[F->index] -= E_f.norm() / (F->centroid - C.centroid).norm();
            }
        }

        /// Contribution from convection term.
        for (int f = 1; f <= N_C; ++f)
        {
            auto curFace = C.surface(f);
            const auto &S_f = C.S(f);

            if (curFace->at_boundary) /// Boundary Case.
            {
                switch (curFace->parent->p_BC)
                {
                case Dirichlet:
                    break; /// When p is given on boundary, p' is 0-value there. Thus, no contribution.
                case Neumann:
                    cur_coef[C.index] += cvec.at(curFace->index-1).dot(S_f); /// When p is not determined on boundary, p' is 0-gradient there.
                    break;
                default:
                    throw unsupported_boundary_condition(curFace->parent->p_BC);
                }
            }
            else /// Internal Case.
            {
                auto F = C.adjCell(f);
                if (!F)
                    throw inconsistent_connectivity("Cell shouldn't be empty!");

                cur_coef[C.index] += 0.5 * cvec.at(curFace->index-1).dot(S_f);
                cur_coef[F->index] += 0.5 * cvec.at(curFace->index-1).dot(S_f);
            }
        }

        /// Record current line.
        /// Convert index to 0-based.
        for (const auto &it : cur_coef)
            coef.emplace_back(C.index - 1, it.first - 1, it.second);
    }

    /// Set reference.
    //coef.remove_if(is_ref_row);
    //coef.emplace_back(ref_cell, ref_cell, 1.0);
}

void calcPressureCorrectionEquationCoef(Eigen::SparseMatrix<Scalar> &A)
{
    std::list<Eigen::Triplet<Scalar>> coef;

    std::vector<Scalar> diag(NumOfCell, 0.0);
    std::vector<Vector> conv(NumOfFace);

    /// Calculate raw triplets
    gen_coef_triplets(coef, diag, conv);

    /// Assemble
    A.setFromTriplets(coef.begin(), coef.end());
}

void calcPressureCorrectionEquationRHS(Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &rhs, double dt)
{
    std::vector<double> p1(NumOfCell, 0.0), p2(NumOfCell, 0.0);

    /// Initialize
    rhs.setZero();

    /// Calculate
    for (const auto &C : cell)
    {
        /// Part1
        Scalar tmp1 = 0.0;
        //tmp1 -= C.volume / dt * (C.rho_next - C.rho) / dt;
        tmp1 -= C.volume / dt * C.delta_mdot;
        p1.at(C.index-1) = tmp1;

        /// Part2
        Scalar tmp2 = 0.0;
        for (int f = 0; f < C.surface.size(); ++f)
        {
            auto curFace = C.surface.at(f);
            const auto &S_f = C.S.at(f);
            const auto &T_f = C.St.at(f);

            /// Raw contribution
            //tmp2 -= (curFace->rhoU_star.dot(S_f)) / dt;

            /// Additional contribution due to cross-diffusion
            if(curFace->at_boundary)
            {
                if(curFace->parent->p_BC == Dirichlet)
                    tmp2 += curFace->grad_p_prime.dot(T_f);
            }
            else
                tmp2 += curFace->grad_p_prime.dot(T_f);
        }
        p2.at(C.index-1) = tmp2;

        /// Gather
        rhs(C.index - 1) = tmp1 + tmp2;
    }

    double p1_min = p1.at(0);
    double p1_max = p1_min;
    for (size_t i = 1; i < NumOfCell; ++i)
    {
        const auto val = p1.at(i);
        if(val < p1_min)
            p1_min = val;
        if(val > p1_max)
            p1_max = val;
    }
    //std::cout << "\np1: " << p1_min << " ~ " << p1_max;
    double p2_min = p2.at(0);
    double p2_max = p2_min;
    for (size_t i = 1; i < NumOfCell; ++i)
    {
        const auto val = p2.at(i);
        if(val < p2_min)
            p2_min = val;
        if(val > p2_max)
            p2_max = val;
    }
    //std::cout << "\np2: " << p2_min << " ~ " << p2_max << std::endl;

    /// Set reference
    //rhs(ref_cell) = ref_val;
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

void calcPressureCorrectionEquationCoef
(
    SX_MAT &B,
    const std::vector<Scalar> &ud,
    const std::vector<Vector> &cvec
)
{
    /// Calculate raw triplets
    std::list<Eigen::Triplet<Scalar>> coef;

    gen_coef_triplets(coef, ud, cvec);

    /// Allocate storage
    B = sx_mat_struct_create(NumOfCell, NumOfCell, coef.size());

    /// Transform
    auto Ai = new SX_INT[B.num_nnzs];
    auto Aj = new SX_INT[B.num_nnzs];
    auto Ax = new SX_FLT[B.num_nnzs];

    int n = 0;
    for (const auto &e : coef)
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

void calcPressureCorrectionEquationRHS(SX_VEC &rhs, double dt)
{
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> x;
    x.resize(NumOfCell, Eigen::NoChange);

    calcPressureCorrectionEquationRHS(x, dt);

    for (SX_INT i = 0; i < NumOfCell; ++i)
        sx_vec_set_entry(&rhs, i, x(i));
}

void prepare_dp_solver(SX_MAT &A, SX_AMG &mg)
{
    SX_AMG_PARS pars;

    sx_amg_pars_init(&pars);
    pars.maxit = 200;
    pars.verb = 0;
    pars.cs_type = SX_COARSE_RS;
    pars.interp_type = SX_INTERP_STD;
    pars.strong_threshold = 0.3;
    pars.trunc_threshold = 0.2;

    //sx_amg_pars_print(&pars);

    sx_amg_setup(&mg, &A, &pars);
}
