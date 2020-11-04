#include "../inc/custom_type.h"
#include "../inc/Gradient.h"

extern int NumOfPnt;
extern int NumOfFace;
extern int NumOfCell;
extern NaturalArray<Point> pnt;
extern NaturalArray<Face> face;
extern NaturalArray<Cell> cell;
extern NaturalArray<Patch> patch;

typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 3> MatX3;
typedef Eigen::Matrix<Scalar, 3, Eigen::Dynamic> Mat3X;
typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatXX;

/// Coefficient matrix used by the least-square method
static std::vector<Mat3X> J_INV_rho;
static std::vector<Mat3X> J_INV_u;
static std::vector<Mat3X> J_INV_v;
static std::vector<Mat3X> J_INV_w;
static std::vector<Mat3X> J_INV_p;
static std::vector<Mat3X> J_INV_T;
static std::vector<Mat3X> J_INV_p_prime;

/**
 * Convert Eigen's intrinsic QR decomposition matrix into R^-1 * Q^T
 * @param J The coefficient matrix to be factorized.
 * @param J_INV The general inverse of input matrix using QR decomposition.
 */
static void extract_qr_matrix(const MatX3 &J, Mat3X &J_INV)
{
    auto QR = J.householderQr();
    const MatXX Q = QR.householderQ();
    const MatXX R = QR.matrixQR().triangularView<Eigen::Upper>();

    const MatXX Q0 = Q.block(0, 0, J.rows(), 3);
    const MatXX R0 = R.block<3, 3>(0, 0);

    J_INV = R0.inverse() * Q0.transpose();
}

/**
 * QR decomposition matrix of each cell.
 */
void prepare_lsq()
{
    MatX3 J_rho;
    std::array<MatX3, 3> J_U;
    MatX3 J_p, J_p_prime;
    MatX3 J_T;

    J_INV_rho.resize(NumOfCell);
    J_INV_u.resize(NumOfCell);
    J_INV_v.resize(NumOfCell);
    J_INV_w.resize(NumOfCell);
    J_INV_p.resize(NumOfCell);
    J_INV_T.resize(NumOfCell);
    J_INV_p_prime.resize(NumOfCell);

    for (auto &c : cell)
    {
        const auto nF = c.surface.size();

        J_rho.resize(nF, Eigen::NoChange);
        J_U[0].resize(nF, Eigen::NoChange);
        J_U[1].resize(nF, Eigen::NoChange);
        J_U[2].resize(nF, Eigen::NoChange);
        J_p.resize(nF, Eigen::NoChange);
        J_p_prime.resize(nF, Eigen::NoChange);
        J_T.resize(nF, Eigen::NoChange);

        for (int j = 0; j < nF; ++j)
        {
            /// Possible coefficients for current face
            auto curFace = c.surface.at(j);
            Vector d, n;
            if (curFace->at_boundary)
            {
                d = curFace->centroid - c.centroid;
                n = c.S.at(j) / curFace->area;

                const Scalar w = 1.0 / d.norm();

                /// Density
                switch (curFace->parent->rho_BC)
                {
                case Dirichlet:
                    J_rho.row(j) << w * d.x(), w * d.y(), w * d.z();
                    break;
                case Neumann:
                    J_rho.row(j) << n.x(), n.y(), n.z();
                    break;
                case Robin:
                    throw robin_bc_is_not_supported();
                default:
                    break;
                }

                /// Velocity-X
                switch (curFace->parent->U_BC[0])
                {
                case Dirichlet:
                    J_U[0].row(j) << w * d.x(), w * d.y(), w * d.z();
                    break;
                case Neumann:
                    J_U[0].row(j) << n.x(), n.y(), n.z();
                    break;
                case Robin:
                    throw robin_bc_is_not_supported();
                default:
                    break;
                }

                /// Velocity-Y
                switch (curFace->parent->U_BC[1])
                {
                case Dirichlet:
                    J_U[1].row(j) << w * d.x(), w * d.y(), w * d.z();
                    break;
                case Neumann:
                    J_U[1].row(j) << n.x(), n.y(), n.z();
                    break;
                case Robin:
                    throw robin_bc_is_not_supported();
                default:
                    break;
                }

                /// Velocity-Z
                switch (curFace->parent->U_BC[2])
                {
                case Dirichlet:
                    J_U[2].row(j) << w * d.x(), w * d.y(), w * d.z();
                    break;
                case Neumann:
                    J_U[2].row(j) << n.x(), n.y(), n.z();
                    break;
                case Robin:
                    throw robin_bc_is_not_supported();
                default:
                    break;
                }

                /// Pressure
                switch (curFace->parent->p_BC)
                {
                case Dirichlet:
                    J_p.row(j) << w * d.x(), w * d.y(), w * d.z();
                    break;
                case Neumann:
                    J_p.row(j) << n.x(), n.y(), n.z();
                    break;
                case Robin:
                    throw robin_bc_is_not_supported();
                default:
                    break;
                }

                /// Pressure-Correction
                switch (curFace->parent->p_prime_BC)
                {
                case Dirichlet:
                    J_p_prime.row(j) << w * d.x(), w * d.y(), w * d.z();
                    break;
                case Neumann:
                    J_p_prime.row(j) << n.x(), n.y(), n.z();
                    break;
                case Robin:
                    throw robin_bc_is_not_supported();
                default:
                    break;
                }

                /// Temperature
                switch (curFace->parent->T_BC)
                {
                case Dirichlet:
                    J_T.row(j) << w * d.x(), w * d.y(), w * d.z();
                    break;
                case Neumann:
                    J_T.row(j) << n.x(), n.y(), n.z();
                    break;
                case Robin:
                    throw robin_bc_is_not_supported();
                default:
                    break;
                }
            }
            else
            {
                auto curAdjCell = c.adjCell.at(j);
                d = curAdjCell->centroid - c.centroid;

                const Scalar w = 1.0 / d.norm();

                /// Density
                J_rho.row(j) << w * d.x(), w * d.y(), w * d.z();

                /// Velocity-X
                J_U[0].row(j) << w * d.x(), w * d.y(), w * d.z();

                /// Velocity-Y
                J_U[1].row(j) << w * d.x(), w * d.y(), w * d.z();

                /// Velocity-Z
                J_U[2].row(j) << w * d.x(), w * d.y(), w * d.z();

                /// Pressure
                J_p.row(j) << w * d.x(), w * d.y(), w * d.z();

                /// Pressure-Correction
                J_p_prime.row(j) << w * d.x(), w * d.y(), w * d.z();

                /// Temperature
                J_T.row(j) << w * d.x(), w * d.y(), w * d.z();
            }
        }

        /// Density
        extract_qr_matrix(J_rho, J_INV_rho.at(c.index - 1));

        /// Velocity-X
        extract_qr_matrix(J_U[0], J_INV_u.at(c.index - 1));

        /// Velocity-Y
        extract_qr_matrix(J_U[1], J_INV_v.at(c.index - 1));

        /// Velocity-Z
        extract_qr_matrix(J_U[2], J_INV_w.at(c.index - 1));

        /// Pressure
        extract_qr_matrix(J_p, J_INV_p.at(c.index - 1));

        /// Temperature
        extract_qr_matrix(J_T, J_INV_T.at(c.index - 1));

        /// Pressure-Correction
        extract_qr_matrix(J_p_prime, J_INV_p_prime.at(c.index - 1));
    }
}

/**
 * Nodal interpolation coefficients regarding to its dependent cells.
 */
void prepare_gg()
{
    for (int i = 1; i <= NumOfPnt; ++i)
    {
        auto &n_dst = pnt(i);
        const int N = n_dst.dependent_cell.size();
        n_dst.cell_weights.resize(N);
        Scalar s = 0.0;
        for (int j = 1; j <= N; ++j)
        {
            auto curAdjCell = n_dst.dependent_cell(j);
            const Scalar weighting = 1.0 / (n_dst.coordinate - curAdjCell->centroid).norm();
            n_dst.cell_weights(j) = weighting;
            s += weighting;
        }
        for (int j = 1; j <= n_dst.cell_weights.size(); ++j)
            n_dst.cell_weights(j) /= s;
    }
}

void calc_cell_primitive_gradient()
{
    for (auto &c : cell)
    {
        const auto nF = c.surface.size();
        Eigen::VectorXd dphi(nF);

        /// density
        for (int i = 0; i < nF; ++i)
        {
            auto curFace = c.surface.at(i);
            if (curFace->at_boundary)
            {
                switch (curFace->parent->rho_BC)
                {
                case Dirichlet:
                    dphi(i) = (curFace->rho - c.rho) / (curFace->centroid - c.centroid).norm();
                    break;
                case Neumann:
                    dphi(i) = curFace->sn_grad_rho;
                    break;
                case Robin:
                    throw robin_bc_is_not_supported();
                default:
                    break;
                }
            }
            else
            {
                auto curAdjCell = c.adjCell.at(i);
                dphi(i) = (curAdjCell->rho - c.rho) / (curAdjCell->centroid - c.centroid).norm();
            }
        }
        c.grad_rho = J_INV_rho.at(c.index - 1) * dphi;

        /// velocity-x
        for (int i = 0; i < nF; ++i)
        {
            auto curFace = c.surface.at(i);
            if (curFace->at_boundary)
            {
                switch (curFace->parent->U_BC[0])
                {
                case Dirichlet:
                    dphi(i) = (curFace->U.x() - c.U.x()) / (curFace->centroid - c.centroid).norm();
                    break;
                case Neumann:
                    dphi(i) = curFace->sn_grad_U.x();
                    break;
                case Robin:
                    throw robin_bc_is_not_supported();
                default:
                    break;
                }
            }
            else
            {
                auto curAdjCell = c.adjCell.at(i);
                dphi(i) = (curAdjCell->U.x() - c.U.x()) / (curAdjCell->centroid - c.centroid).norm();
            }
        }
        c.grad_U.col(0) = J_INV_u.at(c.index - 1) * dphi;

        /// velocity-y
        for (int i = 0; i < nF; ++i)
        {
            auto curFace = c.surface.at(i);
            if (curFace->at_boundary)
            {
                switch (curFace->parent->U_BC[1])
                {
                case Dirichlet:
                    dphi(i) = (curFace->U.y() - c.U.y()) / (curFace->centroid - c.centroid).norm();
                    break;
                case Neumann:
                    dphi(i) = curFace->sn_grad_U.y();
                    break;
                case Robin:
                    throw robin_bc_is_not_supported();
                default:
                    break;
                }
            }
            else
            {
                auto curAdjCell = c.adjCell.at(i);
                dphi(i) = (curAdjCell->U.y() - c.U.y()) / (curAdjCell->centroid - c.centroid).norm();
            }
        }
        c.grad_U.col(1) = J_INV_v.at(c.index - 1) * dphi;

        /// velocity-z
        for (int i = 0; i < nF; ++i)
        {
            auto curFace = c.surface.at(i);
            if (curFace->at_boundary)
            {
                switch (curFace->parent->U_BC[2])
                {
                case Dirichlet:
                    dphi(i) = (curFace->U.z() - c.U.z()) / (curFace->centroid - c.centroid).norm();
                    break;
                case Neumann:
                    dphi(i) = curFace->sn_grad_U.z();
                    break;
                case Robin:
                    throw robin_bc_is_not_supported();
                default:
                    break;
                }
            }
            else
            {
                auto curAdjCell = c.adjCell.at(i);
                dphi(i) = (curAdjCell->U.z() - c.U.z()) / (curAdjCell->centroid - c.centroid).norm();
            }
        }
        c.grad_U.col(2) = J_INV_w.at(c.index - 1) * dphi;

        /// pressure
        for (int i = 0; i < nF; ++i)
        {
            auto curFace = c.surface.at(i);
            if (curFace->at_boundary)
            {
                switch (curFace->parent->p_BC)
                {
                case Dirichlet:
                    dphi(i) = (curFace->p - c.p) / (curFace->centroid - c.centroid).norm();
                    break;
                case Neumann:
                    dphi(i) = curFace->sn_grad_p;
                    break;
                case Robin:
                    throw robin_bc_is_not_supported();
                default:
                    break;
                }
            }
            else
            {
                auto curAdjCell = c.adjCell.at(i);
                dphi(i) = (curAdjCell->p - c.p) / (curAdjCell->centroid - c.centroid).norm();
            }
        }
        c.grad_p = J_INV_p.at(c.index - 1) * dphi;

        /// temperature
        for (int i = 0; i < nF; ++i)
        {
            auto curFace = c.surface.at(i);
            if (curFace->at_boundary)
            {
                switch (curFace->parent->T_BC)
                {
                case Dirichlet:
                    dphi(i) = (curFace->T - c.T) / (curFace->centroid - c.centroid).norm();
                    break;
                case Neumann:
                    dphi(i) = curFace->sn_grad_T;
                    break;
                case Robin:
                    throw robin_bc_is_not_supported();
                default:
                    break;
                }
            }
            else
            {
                auto curAdjCell = c.adjCell.at(i);
                dphi(i) = (curAdjCell->T - c.T) / (curAdjCell->centroid - c.centroid).norm();
            }
        }
        c.grad_T = J_INV_T.at(c.index - 1) * dphi;
    }
}

void calc_cell_pressure_correction_gradient()
{
    for (auto &c : cell)
    {
        const auto nF = c.surface.size();
        Eigen::VectorXd dphi(nF);

        /// p_prime
        for (int i = 0; i < nF; ++i)
        {
            auto curFace = c.surface.at(i);

            if (curFace->at_boundary)
            {
                switch (curFace->parent->p_prime_BC)
                {
                case Dirichlet:
                    dphi(i) = -c.p_prime / (curFace->centroid - c.centroid).norm(); /// Zero-Value is assumed.
                    break;
                case Neumann:
                    dphi(i) = 0.0; /// Zero-Gradient is assumed.
                    break;
                case Robin:
                    throw robin_bc_is_not_supported();
                default:
                    break;
                }
            }
            else
            {
                auto curAdjCell = c.adjCell.at(i);
                dphi(i) = (curAdjCell->p_prime - c.p_prime) / (curAdjCell->centroid - c.centroid).norm();
            }
        }
        c.grad_p_prime = J_INV_p_prime.at(c.index - 1) * dphi;
    }
}

/**
 * Calculate gradient of any scalar "phi" on face from interpolation.
 * @param gpf0 Predicted gradient of "phi" on face.
 * @param phi_C Value of "phi" on local position.
 * @param phi_F Value of "phi" on remote position.
 * @param e_CF Direction from local to remote, normalized to unity.
 * @param d Distance from local to remote.
 * @param dst Interpolation result.
 */
static inline void interpGradientToFace(const Vector &gpf0, Scalar phi_C, Scalar phi_F, const Vector &e_CF, Scalar d, Vector &dst)
{
    dst = gpf0 + ((phi_F - phi_C) / d - gpf0.dot(e_CF)) * e_CF;
}

/**
 * Calculate gradient of any scalar "phi" on face from interpolation.
 * @param gpf0 Predicted gradient of "phi" on face.
 * @param sn_gpf Gradient of "phi" on face in surface normal direction.
 * @param n Surface outward unit normal vector.
 * @param dst Interpolation result.
 */
static inline void interpGradientToFace(const Vector &gpf0, Scalar sn_gpf, const Vector &n, Vector &dst)
{
    dst = gpf0 - gpf0.dot(n) * n + sn_gpf * n;
}

void calc_face_primitive_gradient()
{
    std::array<Vector, 3> gv;

    for (auto &f : face)
    {
        if (f.at_boundary)
        {
            Cell *c;
            bool adj_to_0;
            if (f.c0)
            {
                c = f.c0;
                adj_to_0 = true;
            }
            else if (f.c1)
            {
                c = f.c1;
                adj_to_0 = false;
            }
            else
                throw empty_connectivity(f.index);

            const Vector &r_C = c->centroid;
            const Vector &r_F = f.centroid;
            Vector e_CF = r_F - r_C;
            const Scalar d = e_CF.norm(); /// Distance from cell centroid to face centroid.
            e_CF /= d; /// Unit displacement vector from cell centroid to face centroid.
            const Vector &n = adj_to_0 ? f.n01 : f.n10; /// Unit outward surface normal vector.

            /// density
            switch (f.parent->rho_BC)
            {
            case Dirichlet:
                interpGradientToFace(c->grad_rho, c->rho, f.rho, e_CF, d, f.grad_rho);
                break;
            case Neumann:
                interpGradientToFace(c->grad_rho, f.sn_grad_rho, n, f.grad_rho);
                break;
            case Robin:
                throw robin_bc_is_not_supported();
            default:
                break;
            }

            /// velocity-x
            switch (f.parent->U_BC[0])
            {
            case Dirichlet:
                interpGradientToFace(c->grad_U.col(0), c->U.x(), f.U.x(), e_CF, d, gv[0]);
                break;
            case Neumann:
                interpGradientToFace(c->grad_U.col(0), f.sn_grad_U.x(), n, gv[0]);
                break;
            case Robin:
                throw robin_bc_is_not_supported();
            default:
                break;
            }

            /// velocity-y
            switch (f.parent->U_BC[1])
            {
            case Dirichlet:
                interpGradientToFace(c->grad_U.col(1), c->U.y(), f.U.y(), e_CF, d, gv[1]);
                break;
            case Neumann:
                interpGradientToFace(c->grad_U.col(1), f.sn_grad_U.y(), n, gv[1]);
                break;
            case Robin:
                throw robin_bc_is_not_supported();
            default:
                break;
            }

            /// velocity-z
            switch (f.parent->U_BC[2])
            {
            case Dirichlet:
                interpGradientToFace(c->grad_U.col(2), c->U.z(), f.U.z(), e_CF, d, gv[2]);
                break;
            case Neumann:
                interpGradientToFace(c->grad_U.col(2), f.sn_grad_U.z(), n, gv[2]);
                break;
            case Robin:
                throw robin_bc_is_not_supported();
            default:
                break;
            }

            f.grad_U.col(0) = gv[0];
            f.grad_U.col(1) = gv[1];
            f.grad_U.col(2) = gv[2];

            /// pressure
            switch (f.parent->p_BC)
            {
            case Dirichlet:
                interpGradientToFace(c->grad_p, c->p, f.p, e_CF, d, f.grad_p);
                break;
            case Neumann:
                interpGradientToFace(c->grad_p, f.sn_grad_p, n, f.grad_p);
                break;
            case Robin:
                throw robin_bc_is_not_supported();
            default:
                break;
            }

            /// temperature
            switch (f.parent->T_BC)
            {
            case Dirichlet:
                interpGradientToFace(c->grad_T, c->T, f.T, e_CF, d, f.grad_T);
                break;
            case Neumann:
                interpGradientToFace(c->grad_T, f.sn_grad_T, n, f.grad_T);
                break;
            case Robin:
                throw robin_bc_is_not_supported();
            default:
                break;
            }
        }
        else
        {
            const Vector &r_C = f.c0->centroid;
            const Vector &r_F = f.c1->centroid;
            Vector e_CF = r_F - r_C;
            const Scalar d_CF = e_CF.norm(); /// Distance from local cell centroid to remote cell centroid.
            e_CF /= d_CF; /// Unit vector from local cell "f.c0" to remote cell "f.c1".

            /// density
            const Vector predicted_grad_rho = f.ksi0 * f.c0->grad_rho + f.ksi1 * f.c1->grad_rho;
            const Scalar rho_C = f.c0->rho;
            const Scalar rho_F = f.c1->rho;
            interpGradientToFace(predicted_grad_rho, rho_C, rho_F, e_CF, d_CF, f.grad_rho);

            /// velocity-x
            const Vector predicted_grad_u = f.ksi0 * f.c0->grad_U.col(0) + f.ksi1 * f.c1->grad_U.col(0);
            const Scalar u_C = f.c0->U.x();
            const Scalar u_F = f.c1->U.x();
            interpGradientToFace(predicted_grad_u, u_C, u_F, e_CF, d_CF, gv[0]);

            /// velocity-y
            const Vector predicted_grad_v = f.ksi0 * f.c0->grad_U.col(1) + f.ksi1 * f.c1->grad_U.col(1);
            const Scalar v_C = f.c0->U.y();
            const Scalar v_F = f.c1->U.y();
            interpGradientToFace(predicted_grad_v, v_C, v_F, e_CF, d_CF, gv[1]);

            /// velocity-z
            const Vector predicted_grad_w = f.ksi0 * f.c0->grad_U.col(2) + f.ksi1 * f.c1->grad_U.col(2);
            const Scalar w_C = f.c0->U.z();
            const Scalar w_F = f.c1->U.z();
            interpGradientToFace(predicted_grad_w, w_C, w_F, e_CF, d_CF, gv[2]);

            f.grad_U.col(0) = gv[0];
            f.grad_U.col(1) = gv[1];
            f.grad_U.col(2) = gv[2];

            /// pressure
            const Vector predicted_grad_p = f.ksi0 * f.c0->grad_p + f.ksi1 * f.c1->grad_p;
            const Scalar p_C = f.c0->p;
            const Scalar p_F = f.c1->p;
            interpGradientToFace(predicted_grad_p, p_C, p_F, e_CF, d_CF, f.grad_p);

            /// temperature
            const Vector predicted_grad_T = f.ksi0 * f.c0->grad_T + f.ksi1 * f.c1->grad_T;
            const Scalar T_C = f.c0->T;
            const Scalar T_F = f.c1->T;
            interpGradientToFace(predicted_grad_T, T_C, T_F, e_CF, d_CF, f.grad_T);
        }
    }
}

void calc_face_pressure_correction_gradient()
{
    for (auto &f : face)
    {
        if (f.at_boundary)
        {
            if (f.parent->p_prime_BC == Dirichlet)
            {
                Cell *c = f.c0 ? f.c0 : f.c1;
                Vector d = f.centroid - c->centroid;
                f.grad_p_prime = (-c->p_prime / d.dot(d)) * d; /// 0-value is assumed.
            }
            else if (f.parent->p_prime_BC == Neumann)
                f.grad_p_prime.setZero(); /// 0-gradient in normal direction is assumed.
            else
                throw unsupported_boundary_condition(f.parent->p_prime_BC);
        }
        else
        {
            f.grad_p_prime = f.ksi0 * f.c0->grad_p_prime + f.ksi1 * f.c1->grad_p_prime;
        }
    }
}
