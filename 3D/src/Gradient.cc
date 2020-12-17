#include "../inc/custom_type.h"
#include "../inc/Gradient.h"

extern int NumOfPnt, NumOfFace, NumOfCell;
extern NaturalArray<Point> pnt;
extern NaturalArray<Face> face;
extern NaturalArray<Cell> cell;
extern NaturalArray<Patch> patch;

typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 3> MatX3;
typedef Eigen::Matrix<Scalar, 3, Eigen::Dynamic> Mat3X;
typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatXX;

/// Coefficient matrix used by the least-square method
static std::vector<Mat3X> J_INV_U;
static std::vector<Mat3X> J_INV_p;

/**
 * Convert Eigen's intrinsic QR decomposition matrix into R^-1 * Q^T
 * @param J The coefficient matrix to be factorized.
 * @param J_INV The general inverse of input matrix using QR decomposition.
 */
static void qr_inv(const MatX3 &J, Mat3X &J_INV)
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
    MatX3 J_U;
    MatX3 J_p;

    J_INV_U.resize(NumOfCell);
    J_INV_p.resize(NumOfCell);

    for (auto &c : cell)
    {
        const auto nF = c.surface.size();

        J_U.resize(nF, Eigen::NoChange);
        J_p.resize(nF, Eigen::NoChange);

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

                /// Velocity
                switch (curFace->parent->U_BC)
                {
                case Dirichlet:
                    J_U.row(j) << w * d.x(), w * d.y(), w * d.z();
                    break;
                case Neumann:
                    J_U.row(j) << n.x(), n.y(), n.z();
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
            }
            else
            {
                auto curAdjCell = c.adjCell.at(j);
                d = curAdjCell->centroid - c.centroid;

                const Scalar w = 1.0 / d.norm();

                /// Velocity
                J_U.row(j) << w * d.x(), w * d.y(), w * d.z();

                /// Pressure
                J_p.row(j) << w * d.x(), w * d.y(), w * d.z();
            }
        }

        /// Velocity
        qr_inv(J_U, J_INV_U.at(c.index - 1));

        /// Pressure
        qr_inv(J_p, J_INV_p.at(c.index - 1));
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


void prepare_TeC_operator()
{
    for (auto &c : cell)
    {
        Tensor A;
        A.setZero();
        const size_t Nf = c.surface.size();
        for(size_t j = 0; j < Nf; ++j)
        {
            auto f = c.surface.at(j);
            const Vector &Sf = c.S.at(j);
            A += Sf * Sf.transpose() / f->area;
        }
        c.TeC_INV = A.inverse();
    }
}

void calc_cell_primitive_gradient()
{
    for (auto &c : cell)
    {
        const auto nF = c.surface.size();
        Eigen::VectorXd delta_phi(nF);

        /// velocity-x
        for (int i = 0; i < nF; ++i)
        {
            auto curFace = c.surface.at(i);
            if (curFace->at_boundary)
            {
                switch (curFace->parent->U_BC)
                {
                case Dirichlet:
                    delta_phi(i) = (curFace->U.x() - c.U.x()) / (curFace->centroid - c.centroid).norm();
                    break;
                case Neumann:
                    delta_phi(i) = curFace->sn_grad_U.x();
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
                delta_phi(i) = (curAdjCell->U.x() - c.U.x()) / (curAdjCell->centroid - c.centroid).norm();
            }
        }
        c.grad_U.col(0) = J_INV_U.at(c.index - 1) * delta_phi;

        /// velocity-y
        for (int i = 0; i < nF; ++i)
        {
            auto curFace = c.surface.at(i);
            if (curFace->at_boundary)
            {
                switch (curFace->parent->U_BC)
                {
                case Dirichlet:
                    delta_phi(i) = (curFace->U.y() - c.U.y()) / (curFace->centroid - c.centroid).norm();
                    break;
                case Neumann:
                    delta_phi(i) = curFace->sn_grad_U.y();
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
                delta_phi(i) = (curAdjCell->U.y() - c.U.y()) / (curAdjCell->centroid - c.centroid).norm();
            }
        }
        c.grad_U.col(1) = J_INV_U.at(c.index - 1) * delta_phi;

        /// velocity-z
        for (int i = 0; i < nF; ++i)
        {
            auto curFace = c.surface.at(i);
            if (curFace->at_boundary)
            {
                switch (curFace->parent->U_BC)
                {
                case Dirichlet:
                    delta_phi(i) = (curFace->U.z() - c.U.z()) / (curFace->centroid - c.centroid).norm();
                    break;
                case Neumann:
                    delta_phi(i) = curFace->sn_grad_U.z();
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
                delta_phi(i) = (curAdjCell->U.z() - c.U.z()) / (curAdjCell->centroid - c.centroid).norm();
            }
        }
        c.grad_U.col(2) = J_INV_U.at(c.index - 1) * delta_phi;

        /// pressure
        for (int i = 0; i < nF; ++i)
        {
            auto curFace = c.surface.at(i);
            if (curFace->at_boundary)
            {
                switch (curFace->parent->p_BC)
                {
                case Dirichlet:
                    delta_phi(i) = (curFace->p - c.p) / (curFace->centroid - c.centroid).norm();
                    break;
                case Neumann:
                    delta_phi(i) = curFace->sn_grad_p;
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
                delta_phi(i) = (curAdjCell->p - c.p) / (curAdjCell->centroid - c.centroid).norm();
            }
        }
        c.grad_p = J_INV_p.at(c.index - 1) * delta_phi;
    }
}

Scalar calc_cell_pressure_correction_gradient()
{
    Scalar error = 0.0;
    for (auto &c : cell)
    {
        const auto nF = c.surface.size();
        Eigen::VectorXd delta_phi(nF);

        /// p_prime
        for (int i = 0; i < nF; ++i)
        {
            auto curFace = c.surface.at(i);
            if (curFace->at_boundary)
            {
                switch (curFace->parent->p_BC)
                {
                case Dirichlet:
                    delta_phi(i) = (0.0 - c.p_prime) / (curFace->centroid - c.centroid).norm();
                    break;
                case Neumann:
                    delta_phi(i) = 0.0;
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
                delta_phi(i) = (curAdjCell->p_prime - c.p_prime) / (curAdjCell->centroid - c.centroid).norm();
            }
        }
        const Vector old_gpp = c.grad_p_prime;
        c.grad_p_prime = J_INV_p.at(c.index - 1) * delta_phi;
        error += (c.grad_p_prime - old_gpp).norm();
    }
    error /= NumOfCell;
    return error;
}

void calc_face_primitive_gradient()
{
    std::array<Vector, 3> gv;

    for (auto &f : face)
    {
        if (f.at_boundary)
        {
            auto c = f.c0 ? f.c0 : f.c1;
            const Vector &n = f.c0 ? f.n01 : f.n10;
            const Vector d = f.c0 ? f.r0 : f.r1;
            const Vector alpha = n / n.dot(d);
            const Tensor beta = Tensor::Identity() - alpha * d.transpose();
            const Tensor gamma = Tensor::Identity() - n * n.transpose();

            /// velocity
            if (f.parent->U_BC == Dirichlet)
            {
                gv[0] = alpha * (f.U.x() - c->U.x()) + beta * c->grad_U.col(0);
                gv[1] = alpha * (f.U.y() - c->U.y()) + beta * c->grad_U.col(1);
                gv[2] = alpha * (f.U.z() - c->U.z()) + beta * c->grad_U.col(2);
            }
            else if (f.parent->U_BC == Neumann)
            {
                gv[0] = n * f.sn_grad_U.x() + gamma * c->grad_U.col(0);
                gv[1] = n * f.sn_grad_U.y() + gamma * c->grad_U.col(1);
                gv[2] = n * f.sn_grad_U.z() + gamma * c->grad_U.col(2);
            }
            else
                throw unsupported_boundary_condition(f.parent->U_BC);

            f.grad_U.col(0) = gv[0];
            f.grad_U.col(1) = gv[1];
            f.grad_U.col(2) = gv[2];

            /// pressure
            if (f.parent->p_BC == Dirichlet)
                f.grad_p = alpha * (f.p - c->p) + beta * c->grad_p;
            else if (f.parent->p_BC == Neumann)
                f.grad_p = n * f.sn_grad_p + gamma * c->grad_p;
            else
                throw unsupported_boundary_condition(f.parent->p_BC);
        }
        else
        {
            const Vector d01 = f.c1->centroid - f.c0->centroid;
            const Vector alpha = f.n01 / f.n01.dot(d01);
            const Tensor beta = Tensor::Identity() - alpha * d01.transpose();

            /// velocity
            gv[0] = alpha * (f.c1->U.x() - f.c0->U.x()) + beta * (f.ksi0 * f.c0->grad_U.col(0) + f.ksi1 * f.c1->grad_U.col(0));
            gv[1] = alpha * (f.c1->U.y() - f.c0->U.y()) + beta * (f.ksi0 * f.c0->grad_U.col(1) + f.ksi1 * f.c1->grad_U.col(1));
            gv[2] = alpha * (f.c1->U.z() - f.c0->U.z()) + beta * (f.ksi0 * f.c0->grad_U.col(2) + f.ksi1 * f.c1->grad_U.col(2));

            f.grad_U.col(0) = gv[0];
            f.grad_U.col(1) = gv[1];
            f.grad_U.col(2) = gv[2];

            /// pressure
            f.grad_p = alpha * (f.c1->p - f.c0->p) + beta * (f.ksi0 * f.c0->grad_p + f.ksi1 * f.c1->grad_p);
        }
    }
}

void calc_face_pressure_correction_gradient()
{
    for (auto &f : face)
    {
        if (f.at_boundary)
        {
            const Vector &n = f.c0 ? f.n01 : f.n10;
            auto c = f.c0? f.c0 : f.c1;
            if(f.parent->p_BC == Dirichlet)
            {
                const Vector d = f.centroid - c->centroid;
                f.grad_p_prime = (0.0 - c->p_prime) / d.dot(d) * d;
            }
            else if(f.parent->p_BC == Neumann)
            {
                f.grad_p_prime = (Tensor::Identity() - n * n.transpose()) * c->grad_p_prime;
            }
            else
                throw unsupported_boundary_condition(f.parent->p_BC);
        }
        else
        {
            f.grad_p_prime = f.ksi0 * f.c0->grad_p_prime + f.ksi1 * f.c1->grad_p_prime;
        }
    }
}
