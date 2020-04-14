#include <list>
#include <functional>
#include "../inc/PoissonEqn.h"

extern size_t NumOfPnt, NumOfFace, NumOfCell;
extern NaturalArray<Point> pnt;
extern NaturalArray<Face> face;
extern NaturalArray<Cell> cell;
extern NaturalArray<Patch> patch;

static const size_t ref_cell = 0; // 0-based
static const Scalar ref_val = 0.0; // Pa

bool is_ref_row(const Eigen::Triplet<Scalar> &x)
{
    return x.row() == ref_cell;
}

void calcPressureCorrectionEquationCoef(Eigen::SparseMatrix<Scalar> &A)
{
    std::list<Eigen::Triplet<Scalar>> coef;

    /// Step1: Calculate original coefficients for each cell.
    for (const auto &C : cell)
    {
        /// Initialize coefficient baseline.
        std::map<int, Scalar> cur_coef;
        cur_coef[C.index] = 0.0;
        for (auto F : C.adjCell)
        {
            if (F)
            {
                cur_coef[F->index] = 0.0;
                for (auto FF : F->adjCell)
                {
                    if (FF)
                        cur_coef[FF->index] = 0.0;
                }
            }
        }

        /// Compute coefficient contributions.
        const auto N_C = C.surface.size();
        for (int f = 1; f <= N_C; ++f)
        {
            const auto &S_f = C.S(f);
            auto curFace = C.surface(f);

            if (curFace->atBdry) /// Boundary Case.
            {
                if (curFace->p_prime_BC == Dirichlet)
                {
                    /// When p is given on boundary, dp is 0-value there.
                    const Vector r_C = curFace->center - C.center;
                    const Vector &S_f = C.S(f);
                    cur_coef[C.index] -= r_C.dot(S_f) / r_C.dot(r_C);
                }
                else if (curFace->p_prime_BC == Neumann)
                {
                    /// When p is not determined on boundary, dp is 0-gradient
                    /// there, thus contribution is 0 and no need to handle.
                    continue;
                }
                else
                    throw unsupported_boundary_condition(curFace->p_prime_BC);
            }
            else /// Internal Case.
            {
                auto F = C.adjCell(f);
                if (!F)
                    throw inconsistent_connectivity("Cell shouldn't be empty!");

                const auto N_F = F->surface.size();

                const Vector r_C = curFace->center - C.center;
                const Vector r_F = curFace->center - F->center;
                const Scalar d_f = (F->center - C.center).norm();
                const Vector e_f = (r_C - r_F) / d_f;
                const Scalar ksi_f = 1.0 / (1.0 + r_C.norm() / r_F.norm());
                const Scalar x_f = e_f.dot(S_f);
                const Vector y_f = S_f - x_f * e_f;

                const Eigen::VectorXd J_C = ksi_f * y_f.transpose() * C.J_INV_p;
                const Eigen::VectorXd J_F = (1.0 - ksi_f) * y_f.transpose() * F->J_INV_p;

                /// Part1
                const auto p1_coef = x_f / d_f;
                cur_coef[F->index] += p1_coef;
                cur_coef[C.index] -= p1_coef;

                /// Part2
                for (int i = 0; i < N_C; ++i)
                {
                    auto C_i = C.adjCell.at(i);
                    const auto p2_coef = J_C(i);

                    if (C_i) /// Adjacent cell exists.
                    {
                        cur_coef[C_i->index] += p2_coef;
                        cur_coef[C.index] -= p2_coef;
                    }
                    else /// Adjacent cell does not exist.
                    {
                        const auto localFace = C.surface.at(i);

                        if (localFace->p_prime_BC == Dirichlet)
                        {
                            /// When p is given on boundary, dp is 0-value there.
                            cur_coef[C.index] -= p2_coef;
                        }
                        else if (localFace->p_prime_BC == Neumann)
                        {
                            /// When p is not determined on boundary, dp is 0-gradient
                            /// there, thus contribution of this line is 0.
                            continue;
                        }
                        else
                            throw unsupported_boundary_condition(localFace->p_prime_BC);
                    }
                }

                /// Part3
                for (auto i = 0; i < N_F; ++i)
                {
                    auto F_i = F->adjCell.at(i);
                    const auto p3_coef = J_F(i);

                    if (F_i) /// Adjacent cell exists.
                    {
                        cur_coef[F_i->index] += p3_coef;
                        cur_coef[F->index] -= p3_coef;
                    }
                    else /// Adjacent cell does not exist.
                    {
                        const auto localFace = F->surface.at(i);

                        if (localFace->p_prime_BC == Dirichlet)
                        {
                            /// When p is given on boundary, dp is 0-value there.
                            cur_coef[F->index] -= p3_coef;
                        }
                        else if (localFace->p_prime_BC == Neumann)
                        {
                            /// When p is not determined on boundary, dp is 0-gradient
                            /// there, thus contribution of this line is 0.
                            continue;
                        }
                        else
                            throw unsupported_boundary_condition(localFace->p_prime_BC);
                    }
                }
            }
        }

        /// Record current line.
        /// Convert index to 0-based.
        for (auto it = cur_coef.begin(); it != cur_coef.end(); ++it)
            coef.emplace_back(C.index - 1, it->first - 1, it->second);
    }

    /// Step2: Set reference.
    coef.remove_if(is_ref_row);
    coef.emplace_back(0, 0, 1.0);

    /// Step3: Assemble.
    A.setFromTriplets(coef.begin(), coef.end());
}

void calcPressureCorrectionEquationRHS(Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &rhs)
{
    rhs.setZero();

    for (const auto &C : cell)
    {
        const auto N_C = C.surface.size();
        for (int f = 0; f < N_C; ++f)
        {
            const auto &S_f = C.S.at(f);
            auto curFace = C.surface.at(f);

            rhs(C.index - 1) += curFace->atBdry ? curFace->rhoU.dot(S_f) : curFace->rhoU_star.dot(S_f);
        }
    }

    /// Set reference
    rhs[ref_cell] = ref_val;
}