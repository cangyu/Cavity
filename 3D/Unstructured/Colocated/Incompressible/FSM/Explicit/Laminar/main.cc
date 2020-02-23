#include <iostream>
#include <map>
#include <cmath>
#include <cassert>
#include "xf.h"
#include "custom_type.h"
#include "CHEM.h"
#include "IO.h"
#include "BC.h"
#include "IC.h"


/*************************************************** Global Variables ************************************************/

/* Grid utilities */
size_t NumOfPnt = 0;
size_t NumOfFace = 0;
size_t NumOfCell = 0;

NaturalArray<Point> pnt; // Node objects
NaturalArray<Face> face; // Face objects
NaturalArray<Cell> cell; // Cell objects
NaturalArray<Patch> patch; // Group of boundary faces

/* Pressure-Corrrection equation coefficients */
Eigen::SparseMatrix<Scalar> A_dp;
Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Q_dp;
Eigen::BiCGSTAB<Eigen::SparseMatrix<Scalar>, Eigen::IncompleteLUT<Scalar, int>> dp_solver;


/*********************************************** Least-Squares Method ************************************************/

/**
 * Convert Eigen's intrinsic QR decomposition matrix into R^-1 * Q^T
 * @param J
 * @param J_INV
 */
static void extractQRMatrix(const Eigen::Matrix<Scalar, Eigen::Dynamic, 3> &J, Eigen::Matrix<Scalar, 3, Eigen::Dynamic> &J_INV)
{
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Mat;

    auto QR = J.householderQr();
    const Mat Q = QR.householderQ();
    const Mat R = QR.matrixQR().triangularView<Eigen::Upper>();

    const Mat Q0 = Q.block(0, 0, J.rows(), 3);
    const Mat R0 = R.block<3, 3>(0, 0);

    J_INV = R0.inverse() * Q0.transpose();
}

/**
 * QR decomposition matrix of each cell.
 * Ghost cells are used when the B.C. of boundary faces are set to Neumann type.
 */
void calcLeastSquareCoef()
{
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 3> Mat;

    Mat J_rho;
    std::array<Mat, 3> J_U;
    Mat J_p;
    Mat J_T;

    for (auto &c : cell)
    {
        const size_t nF = c.surface.size();

        J_rho.resize(nF, Eigen::NoChange);
        J_U[0].resize(nF, Eigen::NoChange);
        J_U[1].resize(nF, Eigen::NoChange);
        J_U[2].resize(nF, Eigen::NoChange);
        J_p.resize(nF, Eigen::NoChange);
        J_T.resize(nF, Eigen::NoChange);

        for (size_t j = 0; j < nF; ++j)
        {
            auto curFace = c.surface.at(j);
            if (curFace->atBdry)
            {
                const auto dx = curFace->center.x() - c.center.x();
                const auto dy = curFace->center.y() - c.center.y();
                const auto dz = curFace->center.z() - c.center.z();

                const auto dx2 = 2 * dx;
                const auto dy2 = 2 * dy;
                const auto dz2 = 2 * dz;

                // Density
                switch (curFace->rho_BC)
                {
                case Dirichlet:
                    J_rho.row(j) << dx, dy, dz;
                    break;
                case Neumann:
                    J_rho.row(j) << dx2, dy2, dz2;
                    break;
                case Robin:
                    throw unsupported_boundary_condition(Robin);
                default:
                    break;
                }

                // Velocity-X
                switch (curFace->U_BC[0])
                {
                case Dirichlet:
                    J_U[0].row(j) << dx, dy, dz;
                    break;
                case Neumann:
                    J_U[0].row(j) << dx2, dy2, dz2;
                    break;
                case Robin:
                    throw unsupported_boundary_condition(Robin);
                default:
                    break;
                }

                // Velocity-Y
                switch (curFace->U_BC[1])
                {
                case Dirichlet:
                    J_U[1].row(j) << dx, dy, dz;
                    break;
                case Neumann:
                    J_U[1].row(j) << dx2, dy2, dz2;
                    break;
                case Robin:
                    throw unsupported_boundary_condition(Robin);
                default:
                    break;
                }

                // Velocity-Z
                switch (curFace->U_BC[2])
                {
                case Dirichlet:
                    J_U[2].row(j) << dx, dy, dz;
                    break;
                case Neumann:
                    J_U[2].row(j) << dx2, dy2, dz2;
                    break;
                case Robin:
                    throw unsupported_boundary_condition(Robin);
                default:
                    break;
                }

                // Pressure
                switch (curFace->p_BC)
                {
                case Dirichlet:
                    J_p.row(j) << dx, dy, dz;
                    break;
                case Neumann:
                    J_p.row(j) << dx2, dy2, dz2;
                    break;
                case Robin:
                    throw unsupported_boundary_condition(Robin);
                default:
                    break;
                }

                // Temperature
                switch (curFace->T_BC)
                {
                case Dirichlet:
                    J_T.row(j) << dx, dy, dz;
                    break;
                case Neumann:
                    J_T.row(j) << dx2, dy2, dz2;
                    break;
                case Robin:
                    throw unsupported_boundary_condition(Robin);
                default:
                    break;
                }
            }
            else
            {
                auto curAdjCell = c.adjCell.at(j);

                const auto dx = curAdjCell->center.x() - c.center.x();
                const auto dy = curAdjCell->center.y() - c.center.y();
                const auto dz = curAdjCell->center.z() - c.center.z();

                // Density
                J_rho.row(j) << dx, dy, dz;

                // Velocity-X
                J_U[0].row(j) << dx, dy, dz;

                // Velocity-Y
                J_U[1].row(j) << dx, dy, dz;

                // Velocity-Z
                J_U[2].row(j) << dx, dy, dz;

                // Pressure
                J_p.row(j) << dx, dy, dz;

                // Temperature
                J_T.row(j) << dx, dy, dz;
            }
        }

        // Density
        extractQRMatrix(J_rho, c.J_INV_rho);

        // Velocity-X
        extractQRMatrix(J_U[0], c.J_INV_U[0]);

        // Velocity-Y
        extractQRMatrix(J_U[1], c.J_INV_U[1]);

        // Velocity-Z
        extractQRMatrix(J_U[2], c.J_INV_U[2]);

        // Pressure
        extractQRMatrix(J_p, c.J_INV_p);

        // Temperature
        extractQRMatrix(J_T, c.J_INV_T);
    }
}

void calcPressureCorrectionEquationCoef(Eigen::SparseMatrix<Scalar> &A)
{
    std::vector<Eigen::Triplet<Scalar>> coef;

    for (const auto &C : cell)
    {
        // Initialize coefficient baseline.
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

        // Compute coefficient contributions.
        const auto N_C = C.surface.size();
        for (int f = 1; f <= N_C; ++f)
        {
            const auto &S_f = C.S(f);
            auto curFace = C.surface(f);

            if (curFace->atBdry)
            {
                // No need to handle boundary faces as zero-gradient B.C. is assumed for dp.
                continue;
            }
            else
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

                // Part1
                const auto p1_coef = x_f / d_f;
                cur_coef[F->index] += p1_coef;
                cur_coef[C.index] -= p1_coef;

                // Part2
                for (int i = 0; i < N_C; ++i)
                {
                    auto C_i = C.adjCell.at(i);
                    if (C_i)
                    {
                        const auto p2_coef = J_C(i);
                        cur_coef[C_i->index] += p2_coef;
                        cur_coef[C.index] -= p2_coef;
                    }
                    else
                    {
                        // No need to handle boundary faces as zero-gradient B.C. is assumed for dp.
                        continue;
                    }
                }

                // Part3
                for (auto i = 0; i < N_F; ++i)
                {
                    auto F_i = F->adjCell.at(i);
                    if (F_i)
                    {
                        const auto p3_coef = J_F(i);
                        cur_coef[F_i->index] += p3_coef;
                        cur_coef[F->index] -= p3_coef;
                    }
                    else
                    {
                        // No need to handle boundary faces as zero-gradient B.C. is assumed for dp.
                        continue;
                    }
                }
            }
        }

        // Record current line.
        // Convert index to 0-based.
        for (auto it = cur_coef.begin(); it != cur_coef.end(); ++it)
            coef.emplace_back(C.index - 1, it->first - 1, it->second);
    }

    A.setFromTriplets(coef.begin(), coef.end());
}

/****************************************************** Property *****************************************************/

void calcCellProperty()
{
    for (auto &c : cell)
    {
        // Dynamic viscosity
        c.mu = Sutherland(c.T);
    }
}

void calcFaceProperty()
{
    for (auto &f : face)
    {
        // Dynamic viscosity
        f.mu = Sutherland(f.T);
    }
}

/*********************************************** Spatial Discretization **********************************************/

void calcFaceGhostVariable()
{
    for (auto &f : face)
    {
        if (f.atBdry)
        {
            if (f.c0)
            {
                // density
                switch (f.rho_BC)
                {
                case Neumann:
                    f.rho_ghost = f.c0->rho + 2 * f.grad_rho.dot(f.r0);
                    break;
                case Dirichlet:
                    f.rho_ghost = 0.0;
                    break;
                case Robin:
                    throw unsupported_boundary_condition(Robin);
                default:
                    break;
                }

                // velocity-x
                switch (f.U_BC[0])
                {
                case Neumann:
                    f.U_ghost.x() = f.c0->U[0] + 2 * f.grad_U.col(0).dot(f.r0);
                    break;
                case Dirichlet:
                    f.U_ghost.x() = 0.0;
                    break;
                case Robin:
                    throw unsupported_boundary_condition(Robin);
                default:
                    break;
                }

                // velocity-y
                switch (f.U_BC[1])
                {
                case Neumann:
                    f.U_ghost.y() = f.c0->U[1] + 2 * f.grad_U.col(1).dot(f.r0);
                    break;
                case Dirichlet:
                    f.U_ghost.y() = 0.0;
                    break;
                case Robin:
                    throw unsupported_boundary_condition(Robin);
                default:
                    break;
                }

                // velocity-z
                switch (f.U_BC[2])
                {
                case Neumann:
                    f.U_ghost.z() = f.c0->U[2] + 2 * f.grad_U.col(2).dot(f.r0);
                    break;
                case Dirichlet:
                    f.U_ghost.z() = 0.0;
                    break;
                case Robin:
                    throw unsupported_boundary_condition(Robin);
                default:
                    break;
                }

                // pressure
                switch (f.p_BC)
                {
                case Neumann:
                    f.p_ghost = f.c0->p + 2 * f.grad_p.dot(f.r0);
                    break;
                case Dirichlet:
                    f.p_ghost = 0.0;
                    break;
                case Robin:
                    throw unsupported_boundary_condition(Robin);
                default:
                    break;
                }

                // temperature
                switch (f.T_BC)
                {
                case Neumann:
                    f.T_ghost = f.c0->T + 2 * f.grad_T.dot(f.r0);
                    break;
                case Dirichlet:
                    f.p_ghost = 0.0;
                    break;
                case Robin:
                    throw unsupported_boundary_condition(Robin);
                default:
                    break;
                }
            }
            else if (f.c1)
            {
                // density
                switch (f.rho_BC)
                {
                case Neumann:
                    f.rho_ghost = f.c1->rho + 2 * f.grad_rho.dot(f.r1);
                    break;
                case Dirichlet:
                    f.rho_ghost = 0.0;
                    break;
                case Robin:
                    throw unsupported_boundary_condition(Robin);
                default:
                    break;
                }

                // velocity-x
                switch (f.U_BC[0])
                {
                case Neumann:
                    f.U_ghost.x() = f.c1->U[0] + 2 * f.grad_U.col(0).dot(f.r1);
                    break;
                case Dirichlet:
                    f.U_ghost.x() = 0.0;
                    break;
                case Robin:
                    throw unsupported_boundary_condition(Robin);
                default:
                    break;
                }

                // velocity-y
                switch (f.U_BC[1])
                {
                case Neumann:
                    f.U_ghost.y() = f.c1->U[1] + 2 * f.grad_U.col(1).dot(f.r1);
                    break;
                case Dirichlet:
                    f.U_ghost.y() = 0.0;
                    break;
                case Robin:
                    throw unsupported_boundary_condition(Robin);
                default:
                    break;
                }

                // velocity-z
                switch (f.U_BC[2])
                {
                case Neumann:
                    f.U_ghost.z() = f.c1->U[2] + 2 * f.grad_U.col(2).dot(f.r1);
                    break;
                case Dirichlet:
                    f.U_ghost.z() = 0.0;
                    break;
                case Robin:
                    throw unsupported_boundary_condition(Robin);
                default:
                    break;
                }

                // pressure
                switch (f.p_BC)
                {
                case Neumann:
                    f.p_ghost = f.c1->p + 2 * f.grad_p.dot(f.r1);
                    break;
                case Dirichlet:
                    f.p_ghost = 0.0;
                    break;
                case Robin:
                    throw unsupported_boundary_condition(Robin);
                default:
                    break;
                }

                // temperature
                switch (f.T_BC)
                {
                case Neumann:
                    f.T_ghost = f.c1->T + 2 * f.grad_T.dot(f.r1);
                    break;
                case Dirichlet:
                    f.T_ghost = 0.0;
                    break;
                case Robin:
                    throw unsupported_boundary_condition(Robin);
                default:
                    break;
                }
            }
            else
                throw empty_connectivity(f.index);
        }
    }
}

void calcCellGradient()
{
    for (auto &c : cell)
    {
        const size_t nF = c.surface.size();
        Eigen::VectorXd dphi(nF);

        /* Gradient of density */
        for (size_t i = 0; i < nF; ++i)
        {
            auto curFace = c.surface.at(i);
            auto curAdjCell = c.adjCell.at(i);

            if (curFace->atBdry)
            {
                if (curFace->rho_BC == Dirichlet)
                    dphi(i) = curFace->rho - c.rho;
                else
                    dphi(i) = curFace->rho_ghost - c.rho;
            }
            else
                dphi(i) = curAdjCell->rho - c.rho;
        }
        c.grad_rho = c.J_INV_rho * dphi;

        /* Gradient of x-dim velocity */
        for (size_t i = 0; i < nF; ++i)
        {
            auto curFace = c.surface.at(i);
            auto curAdjCell = c.adjCell.at(i);

            if (curFace->atBdry)
            {
                if (curFace->U_BC[0] == Dirichlet)
                    dphi(i) = curFace->U.x() - c.U.x();
                else
                    dphi(i) = curFace->U_ghost.x() - c.U.x();
            }
            else
                dphi(i) = curAdjCell->U.x() - c.U.x();
        }
        c.grad_U.col(0) = c.J_INV_U[0] * dphi;

        /* Gradient of y-dim velocity */
        for (size_t i = 0; i < nF; ++i)
        {
            auto curFace = c.surface.at(i);
            auto curAdjCell = c.adjCell.at(i);

            if (curFace->atBdry)
            {
                if (curFace->U_BC[1] == Dirichlet)
                    dphi(i) = curFace->U.y() - c.U.y();
                else
                    dphi(i) = curFace->U_ghost.y() - c.U.y();
            }
            else
                dphi(i) = curAdjCell->U.y() - c.U.y();
        }
        c.grad_U.col(1) = c.J_INV_U[1] * dphi;

        /* Gradient of z-dim velocity */
        for (size_t i = 0; i < nF; ++i)
        {
            auto curFace = c.surface.at(i);
            auto curAdjCell = c.adjCell.at(i);

            if (curFace->atBdry)
            {
                if (curFace->U_BC[2] == Dirichlet)
                    dphi(i) = curFace->U.z() - c.U.z();
                else
                    dphi(i) = curFace->U_ghost.z() - c.U.z();
            }
            else
                dphi(i) = curAdjCell->U.z() - c.U.z();
        }
        c.grad_U.col(2) = c.J_INV_U[2] * dphi;

        /* Gradient of pressure */
        for (size_t i = 0; i < nF; ++i)
        {
            auto curFace = c.surface.at(i);
            auto curAdjCell = c.adjCell.at(i);

            if (curFace->atBdry)
            {
                if (curFace->p_BC == Dirichlet)
                    dphi(i) = curFace->p - c.p;
                else
                    dphi(i) = curFace->p_ghost - c.p;
            }
            else
                dphi(i) = curAdjCell->p - c.p;
        }
        c.grad_p = c.J_INV_p * dphi;

        /* Gradient of temperature */
        for (size_t i = 0; i < nF; ++i)
        {
            auto curFace = c.surface.at(i);
            auto curAdjCell = c.adjCell.at(i);

            if (curFace->atBdry)
            {
                if (curFace->T_BC == Dirichlet)
                    dphi(i) = curFace->T - c.T;
                else
                    dphi(i) = curFace->T_ghost - c.T;
            }
            else
                dphi(i) = curAdjCell->T - c.T;
        }
        c.grad_T = c.J_INV_T * dphi;
    }
}

inline Vector interpGradientToFace(const Vector &predicted_grad_phi_f, Scalar phi_C, Scalar phi_F, const Vector &e_CF, Scalar d_CF)
{
    return predicted_grad_phi_f + ((phi_F - phi_C) / d_CF - predicted_grad_phi_f.dot(e_CF))*e_CF;
}

void calcFaceGradient()
{
    for (auto &f : face)
    {
        if (f.atBdry)
        {
            /* Gradients at boundary face */
            if (f.c0)
            {
                const Vector &r_C = f.c0->center;
                const Vector &r_F = f.center;
                Vector e_CF = r_F - r_C;
                const Scalar d_CF = e_CF.norm();
                e_CF /= d_CF;

                // density
                switch (f.rho_BC)
                {
                case Dirichlet:
                    f.grad_rho = interpGradientToFace(f.c0->grad_rho, f.c0->rho, f.rho, e_CF, d_CF);
                    break;
                case Neumann:
                    break;
                case Robin:
                    throw unsupported_boundary_condition(Robin);
                default:
                    break;
                }

                // velocity-x
                switch (f.U_BC[0])
                {
                case Dirichlet:
                    f.grad_U.col(0) = interpGradientToFace(f.c0->grad_U.col(0), f.c0->U.x(), f.U.x(), e_CF, d_CF);
                    break;
                case Neumann:
                    break;
                case Robin:
                    throw unsupported_boundary_condition(Robin);
                default:
                    break;
                }

                // velocity-y
                switch (f.U_BC[1])
                {
                case Dirichlet:
                    f.grad_U.col(1) = interpGradientToFace(f.c0->grad_U.col(1), f.c0->U.y(), f.U.y(), e_CF, d_CF);
                    break;
                case Neumann:
                    break;
                case Robin:
                    throw unsupported_boundary_condition(Robin);
                default:
                    break;
                }

                // velocity-z
                switch (f.U_BC[2])
                {
                case Dirichlet:
                    f.grad_U.col(2) = interpGradientToFace(f.c0->grad_U.col(2), f.c0->U.z(), f.U.z(), e_CF, d_CF);
                    break;
                case Neumann:
                    break;
                case Robin:
                    throw unsupported_boundary_condition(Robin);
                default:
                    break;
                }

                // pressure
                switch (f.p_BC)
                {
                case Dirichlet:
                    f.grad_p = interpGradientToFace(f.c0->grad_p, f.c0->p, f.p, e_CF, d_CF);
                    break;
                case Neumann:
                    break;
                case Robin:
                    throw unsupported_boundary_condition(Robin);
                default:
                    break;
                }

                // temperature
                switch (f.T_BC)
                {
                case Dirichlet:
                    f.grad_T = interpGradientToFace(f.c0->grad_T, f.c0->T, f.T, e_CF, d_CF);
                    break;
                case Neumann:
                    break;
                case Robin:
                    throw unsupported_boundary_condition(Robin);
                default:
                    break;
                }
            }
            else if (f.c1)
            {
                const Vector &r_C = f.c1->center;
                const Vector &r_F = f.center;
                Vector e_CF = r_F - r_C;
                const Scalar d_CF = e_CF.norm();
                e_CF /= d_CF;

                // density
                switch (f.rho_BC)
                {
                case Dirichlet:
                    f.grad_rho = interpGradientToFace(f.c1->grad_rho, f.c1->rho, f.rho, e_CF, d_CF);
                    break;
                case Neumann:
                    break;
                case Robin:
                    throw unsupported_boundary_condition(Robin);
                default:
                    break;
                }

                // velocity-x
                switch (f.U_BC[0])
                {
                case Dirichlet:
                    f.grad_U.col(0) = interpGradientToFace(f.c1->grad_U.col(0), f.c1->U.x(), f.U.x(), e_CF, d_CF);
                    break;
                case Neumann:
                    break;
                case Robin:
                    throw unsupported_boundary_condition(Robin);
                default:
                    break;
                }

                // velocity-y
                switch (f.U_BC[1])
                {
                case Dirichlet:
                    f.grad_U.col(1) = interpGradientToFace(f.c1->grad_U.col(1), f.c1->U.y(), f.U.y(), e_CF, d_CF);
                    break;
                case Neumann:
                    break;
                case Robin:
                    throw unsupported_boundary_condition(Robin);
                default:
                    break;
                }

                // velocity-z
                switch (f.U_BC[2])
                {
                case Dirichlet:
                    f.grad_U.col(2) = interpGradientToFace(f.c1->grad_U.col(2), f.c1->U.z(), f.U.z(), e_CF, d_CF);
                    break;
                case Neumann:
                    break;
                case Robin:
                    throw unsupported_boundary_condition(Robin);
                default:
                    break;
                }

                // pressure
                switch (f.p_BC)
                {
                case Dirichlet:
                    f.grad_p = interpGradientToFace(f.c1->grad_p, f.c1->p, f.p, e_CF, d_CF);
                    break;
                case Neumann:
                    break;
                case Robin:
                    throw unsupported_boundary_condition(Robin);
                default:
                    break;
                }

                // temperature
                switch (f.T_BC)
                {
                case Dirichlet:
                    f.grad_T = interpGradientToFace(f.c1->grad_T, f.c1->T, f.T, e_CF, d_CF);
                    break;
                case Neumann:
                    break;
                case Robin:
                    throw unsupported_boundary_condition(Robin);
                default:
                    break;
                }
            }
            else
                throw empty_connectivity(f.index);
        }
        else
        {
            /* Gradients at internal face */
            const Vector &r_C = f.c0->center;
            const Vector &r_F = f.c1->center;
            Vector e_CF = r_F - r_C;
            const Scalar d_CF = e_CF.norm();
            e_CF /= d_CF;

            const Scalar ksi = f.r1.norm() / (f.r0.norm() + f.r1.norm());

            // density
            const Vector predicted_grad_rho = ksi * f.c0->grad_rho + (1 - ksi) * f.c1->grad_rho;
            const Scalar rho_C = f.c0->rho;
            const Scalar rho_F = f.c1->rho;
            f.grad_rho = interpGradientToFace(predicted_grad_rho, rho_C, rho_F, e_CF, d_CF);

            // velocity-x
            const Vector predicted_grad_u = ksi * f.c0->grad_U.col(0) + (1 - ksi) * f.c1->grad_U.col(0);
            const Scalar u_C = f.c0->U.x();
            const Scalar u_F = f.c1->U.x();
            f.grad_U.col(0) = interpGradientToFace(predicted_grad_u, u_C, u_F, e_CF, d_CF);

            // velocity-y
            const Vector predicted_grad_v = ksi * f.c0->grad_U.col(1) + (1 - ksi) * f.c1->grad_U.col(1);
            const Scalar v_C = f.c0->U.y();
            const Scalar v_F = f.c1->U.y();
            f.grad_U.col(1) = interpGradientToFace(predicted_grad_v, v_C, v_F, e_CF, d_CF);

            // velocity-z
            const Vector predicted_grad_w = ksi * f.c0->grad_U.col(2) + (1 - ksi) * f.c1->grad_U.col(2);
            const Scalar w_C = f.c0->U.z();
            const Scalar w_F = f.c1->U.z();
            f.grad_U.col(2) = interpGradientToFace(predicted_grad_w, w_C, w_F, e_CF, d_CF);

            // pressure
            const Vector predicted_grad_p = ksi * f.c0->grad_p + (1 - ksi) * f.c1->grad_p;
            const Scalar p_C = f.c0->p;
            const Scalar p_F = f.c1->p;
            f.grad_p = interpGradientToFace(predicted_grad_p, p_C, p_F, e_CF, d_CF);

            // temperature
            const Vector predicted_grad_T = ksi * f.c0->grad_T + (1 - ksi) * f.c1->grad_T;
            const Scalar T_C = f.c0->T;
            const Scalar T_F = f.c1->T;
            f.grad_T = interpGradientToFace(predicted_grad_T, T_C, T_F, e_CF, d_CF);
        }
    }
}

void calcFaceValue()
{
    for (auto &f : face)
    {
        if (f.atBdry)
        {
            if (f.c0)
            {
                // density
                switch (f.rho_BC)
                {
                case Dirichlet:
                    break;
                case Neumann:
                    f.rho = f.c0->rho + (f.grad_rho + f.c0->grad_rho).dot(f.r0) / 2;
                    break;
                case Robin:
                    throw unsupported_boundary_condition(Robin);
                default:
                    break;
                }

                // velocity-x
                switch (f.U_BC[0])
                {
                case Dirichlet:
                    break;
                case Neumann:
                    f.U.x() = f.c0->U.x() + (f.grad_U.col(0) + f.c0->grad_U.col(0)).dot(f.r0) / 2;
                    break;
                case Robin:
                    throw unsupported_boundary_condition(Robin);
                default:
                    break;
                }

                // velocity-y
                switch (f.U_BC[1])
                {
                case Dirichlet:
                    break;
                case Neumann:
                    f.U.y() = f.c0->U.y() + (f.grad_U.col(1) + f.c0->grad_U.col(1)).dot(f.r0) / 2;
                    break;
                case Robin:
                    throw unsupported_boundary_condition(Robin);
                default:
                    break;
                }

                // velocity-z
                switch (f.U_BC[2])
                {
                case Dirichlet:
                    break;
                case Neumann:
                    f.U.z() = f.c0->U.z() + (f.grad_U.col(2) + f.c0->grad_U.col(2)).dot(f.r0) / 2;
                    break;
                case Robin:
                    throw unsupported_boundary_condition(Robin);
                default:
                    break;
                }

                // pressure
                switch (f.p_BC)
                {
                case Dirichlet:
                    break;
                case Neumann:
                    f.p = f.c0->p + (f.grad_p + f.c0->grad_p).dot(f.r0) / 2;
                    break;
                case Robin:
                    throw unsupported_boundary_condition(Robin);
                default:
                    break;
                }

                // temperature
                switch (f.T_BC)
                {
                case Dirichlet:
                    break;
                case Neumann:
                    f.T = f.c0->T + (f.grad_T + f.c0->grad_T).dot(f.r0) / 2;
                    break;
                case Robin:
                    throw unsupported_boundary_condition(Robin);
                default:
                    break;
                }
            }
            else if (f.c1)
            {
                // density
                switch (f.rho_BC)
                {
                case Dirichlet:
                    break;
                case Neumann:
                    f.rho = f.c1->rho + (f.grad_rho + f.c1->grad_rho).dot(f.r1) / 2;
                    break;
                case Robin:
                    throw unsupported_boundary_condition(Robin);
                default:
                    break;
                }

                // velocity-x
                switch (f.U_BC[0])
                {
                case Dirichlet:
                    break;
                case Neumann:
                    f.U.x() = f.c1->U.x() + (f.grad_U.col(0) + f.c1->grad_U.col(0)).dot(f.r1) / 2;
                    break;
                case Robin:
                    throw unsupported_boundary_condition(Robin);
                default:
                    break;
                }

                // velocity-y
                switch (f.U_BC[1])
                {
                case Dirichlet:
                    break;
                case Neumann:
                    f.U.y() = f.c1->U.y() + (f.grad_U.col(1) + f.c1->grad_U.col(1)).dot(f.r1) / 2;
                    break;
                case Robin:
                    throw unsupported_boundary_condition(Robin);
                default:
                    break;
                }

                // velocity-z
                switch (f.U_BC[2])
                {
                case Dirichlet:
                    break;
                case Neumann:
                    f.U.z() = f.c1->U.z() + (f.grad_U.col(2) + f.c1->grad_U.col(2)).dot(f.r1) / 2;
                    break;
                case Robin:
                    throw unsupported_boundary_condition(Robin);
                default:
                    break;
                }

                // pressure
                switch (f.p_BC)
                {
                case Dirichlet:
                    break;
                case Neumann:
                    f.p = f.c1->p + (f.grad_p + f.c1->grad_p).dot(f.r1) / 2;
                    break;
                case Robin:
                    throw unsupported_boundary_condition(Robin);
                default:
                    break;
                }

                // temperature
                switch (f.T_BC)
                {
                case Dirichlet:
                    break;
                case Neumann:
                    f.T = f.c1->T + (f.grad_T + f.c1->grad_T).dot(f.r1) / 2;
                    break;
                case Robin:
                    throw unsupported_boundary_condition(Robin);
                default:
                    break;
                }
            }
            else
                throw empty_connectivity(f.index);
        }
        else
        {
            // weighting coefficient
            const Scalar ksi = f.r1.norm() / (f.r0.norm() + f.r1.norm());

            // pressure
            const Scalar p_0 = f.c0->p + f.c0->grad_p.dot(f.r0);
            const Scalar p_1 = f.c1->p + f.c1->grad_p.dot(f.r1);
            f.p = 0.5 * (p_0 + p_1);

            // temperature
            const Scalar T_0 = f.c0->T + f.c0->grad_T.dot(f.r0);
            const Scalar T_1 = f.c1->T + f.c1->grad_T.dot(f.r1);
            f.T = ksi * T_0 + (1.0 - ksi) * T_1;

            // velocity
            if (f.U.dot(f.n01) > 0)
            {
                const Scalar u_0 = f.c0->U.x() + f.c0->grad_U.col(0).dot(f.r0);
                const Scalar v_0 = f.c0->U.y() + f.c0->grad_U.col(1).dot(f.r0);
                const Scalar w_0 = f.c0->U.z() + f.c0->grad_U.col(2).dot(f.r0);

                f.U = { u_0, v_0, w_0 };
            }
            else
            {
                const Scalar u_1 = f.c1->U.x() + f.c1->grad_U.col(0).dot(f.r1);
                const Scalar v_1 = f.c1->U.y() + f.c1->grad_U.col(1).dot(f.r1);
                const Scalar w_1 = f.c1->U.z() + f.c1->grad_U.col(2).dot(f.r1);

                f.U = { u_1, v_1, w_1 };
            }

            // density
            if (f.U.dot(f.n01) > 0)
                f.rho = f.c0->rho + f.c0->grad_rho.dot(f.r0);
            else
                f.rho = f.c1->rho + f.c1->grad_rho.dot(f.r1);
        }
    }
}

void calcCellContinuityFlux()
{
    // TODO
}

void calcCellMomentumFlux()
{
    for (auto &c : cell)
    {
        c.pressure_flux.setZero();
        c.convection_flux.setZero();
        c.viscous_flux.setZero();

        for (size_t j = 0; j < c.S.size(); ++j)
        {
            auto f = c.surface.at(j);
            const auto &Sf = c.S.at(j);

            // convection term
            const Vector cur_convection_flux = f->rhoU * f->U.dot(Sf);
            c.convection_flux += cur_convection_flux;

            // pressure term
            const Vector cur_pressure_flux = f->p * Sf;
            c.pressure_flux += cur_pressure_flux;

            // viscous term
            const Vector cur_viscous_flux = { Sf.dot(f->tau.col(0)), Sf.dot(f->tau.col(1)), Sf.dot(f->tau.col(2)) };
            c.viscous_flux += cur_viscous_flux;
        }
    }
}

void calcCellEnergyFlux()
{
    // TODO
}

void calcCellFlux()
{
    // Continuity equation
    calcCellContinuityFlux();

    // Momentum equation
    calcCellMomentumFlux();

    // Energy equation
    calcCellEnergyFlux();
}

void calcInternalFace_rhoU_star()
{
    for (auto &f : face)
    {
        if (!f.atBdry)
        {
            f.rhoU_star = (f.c0->rhoU_star + f.c1->rhoU0) / 2;
        }
    }
}

void calcPoissonEquationRHS(Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &rhs)
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
}

void calcPressureCorrectionGradient()
{
    for (auto &c : cell)
    {
        const size_t nF = c.surface.size();
        Eigen::VectorXd dphi(nF);

        for (size_t i = 0; i < nF; ++i)
        {
            auto curFace = c.surface.at(i);
            auto curAdjCell = c.adjCell.at(i);

            if (curFace->atBdry)
                dphi(i) = 0.0;
            else
                dphi(i) = curAdjCell->p_prime - c.p_prime;
        }
        c.grad_p_prime = c.J_INV_p_prime * dphi;
    }
}

/*********************************************** Temporal Discretization *********************************************/

/**
 * Explicit Fractional-Step Method.
 * @param TimeStep
 */
void FSM(Scalar TimeStep)
{
    BC();

    // Physical properties at centroid of each cell
    calcCellProperty();

    // Boundary ghost values if any
    calcFaceGhostVariable();

    // Gradients at centroid of each cell
    calcCellGradient();

    // Gradients at centroid of each face
    calcFaceGradient();

    // Interpolate values on each face
    calcFaceValue();

    // Physical properties at centroid of each face
    calcFaceProperty();

    calcCellFlux();

    // Prediction
    for (auto &c : cell)
        c.rhoU_star = c.rhoU0 + TimeStep / c.volume * (-c.convection_flux - c.pressure_flux + c.viscous_flux);

    // rhoU_star at each face
    calcInternalFace_rhoU_star();

    // Correction
    calcPoissonEquationRHS(Q_dp);
    Q_dp /= TimeStep;
    Eigen::VectorXd dp = dp_solver.solve(Q_dp);
    for (int i = 0; i < NumOfCell; ++i)
    {
        auto &c = cell.at(i);
        c.p_prime = dp(i);
    }
    calcPressureCorrectionGradient();

    // Update
    for (int i = 0; i < NumOfCell; ++i)
    {
        auto &c = cell.at(i);

        c.p += c.p_prime;
        c.U = (c.rhoU_star - TimeStep * c.grad_p_prime) / c.rho;

        c.continuity_res = 0.0;
        c.energy_res = 0.0;
    }
}

/**
 * Forward Euler time-marching.
 * @param TimeStep
 */
void Euler(Scalar TimeStep)
{
    /* Init */
    for (size_t i = 1; i <= NumOfCell; ++i)
    {
        auto &c = cell(i);
        c.rho = c.rho0;
        c.U = c.U0;
        c.p = c.p0;
        c.T = c.T0;
    }

    /* Pressure-Velocity coupling */
    FSM(TimeStep);

    /* Update */
    for (size_t i = 1; i <= NumOfCell; ++i)
    {
        auto &c = cell(i);

        c.rho = c.rho0 + TimeStep * c.continuity_res;
        c.T = c.T0 + TimeStep * c.energy_res;

        c.rho0 = c.rho;
        c.U0 = c.U;
        c.p0 = c.p;
        c.T0 = c.T;
        c.rhoU0 = c.rho0 * c.U0;
    }
}

/***************************************************** Solution Control **********************************************/

/* Iteration timing and counting */
const int MAX_ITER = 2000;
const Scalar MAX_TIME = 100.0; // s

bool diagnose()
{
    bool ret = false;

    return ret;
}

Scalar calcTimeStep()
{
    Scalar ret = 1e-5;

    return ret;
}

void solve(std::ostream &fout = std::cout)
{
    static const size_t OUTPUT_GAP = 100;

    int iter = 0;
    Scalar dt = 0.0; // s
    Scalar t = 0.0; // s
    bool done = false;
    while (!done)
    {
        fout << "Iter" << ++iter << ":" << std::endl;
        dt = calcTimeStep();
        fout << "\tt=" << t << "s, dt=" << dt << "s" << std::endl;
        Euler(dt);
        t += dt;
        done = diagnose();
        if (done || !(iter % OUTPUT_GAP))
        {
            updateNodalValue();
            writeTECPLOT_Nodal("flow" + std::to_string(iter) + "_NODAL.dat", "3D Cavity");
            writeTECPLOT_CellCentered("flow" + std::to_string(iter) + "_CELL.dat", "3D Cavity");
        }
    }
    fout << "Finished!" << std::endl;
}

/**
 * Initialize the computation environment.
 */
void init()
{
    // Load mesh.
    readMESH("cube32.msh");

    // Set B.C. of each variable.
    BC_TABLE();

    // Least-Square coefficients used to calculate gradients.
    calcLeastSquareCoef();

    // Pressure-Correction equation coefficients.
    A_dp.resize(NumOfCell, NumOfCell);
    Q_dp.resize(NumOfCell, Eigen::NoChange);
    calcPressureCorrectionEquationCoef(A_dp);
    A_dp.makeCompressed();
    dp_solver.compute(A_dp);

    // Set I.C. of each variable.
    IC();
}

/**
 * Solver entrance.
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, char *argv[])
{
    init();
    solve();

    return 0;
}

/********************************************************* END *******************************************************/
