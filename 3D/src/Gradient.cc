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
static std::vector<Mat3X> J_INV_rho;
static std::vector<Mat3X> J_INV_U;
static std::vector<Mat3X> J_INV_p;
static std::vector<Mat3X> J_INV_T;

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
    MatX3 J_rho;
    MatX3 J_U;
    MatX3 J_p;
    MatX3 J_T;

    J_INV_rho.resize(NumOfCell);
    J_INV_U.resize(NumOfCell);
    J_INV_p.resize(NumOfCell);
    J_INV_T.resize(NumOfCell);

    for (auto &c : cell)
    {
        const auto nF = c.surface.size();
        J_rho.resize(nF, Eigen::NoChange);
        J_U.resize(nF, Eigen::NoChange);
        J_p.resize(nF, Eigen::NoChange);
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
                J_rho.row(j) << w * d.x(), w * d.y(), w * d.z();

                /// Velocity
                const auto U_BC = curFace->parent->U_BC;
                if (U_BC == Dirichlet)
                    J_U.row(j) << w * d.x(), w * d.y(), w * d.z();
                else if (U_BC == Neumann)
                    J_U.row(j) << n.x(), n.y(), n.z();
                else
                    throw unsupported_boundary_condition(U_BC);

                /// Pressure
                const auto p_BC = curFace->parent->p_BC;
                if (p_BC == Dirichlet)
                    J_p.row(j) << w * d.x(), w * d.y(), w * d.z();
                else if (p_BC == Neumann)
                    J_p.row(j) << n.x(), n.y(), n.z();
                else
                    throw unsupported_boundary_condition(p_BC);

                /// Temperature
                const auto T_BC = curFace->parent->T_BC;
                if (T_BC == Dirichlet)
                    J_T.row(j) << w * d.x(), w * d.y(), w * d.z();
                else if (T_BC == Neumann)
                    J_T.row(j) << n.x(), n.y(), n.z();
                else
                    throw unsupported_boundary_condition(T_BC);
            }
            else
            {
                auto curAdjCell = c.adjCell.at(j);
                d = curAdjCell->centroid - c.centroid;

                const Scalar w = 1.0 / d.norm();

                /// Density
                J_rho.row(j) << w * d.x(), w * d.y(), w * d.z();

                /// Velocity
                J_U.row(j) << w * d.x(), w * d.y(), w * d.z();

                /// Pressure
                J_p.row(j) << w * d.x(), w * d.y(), w * d.z();

                /// Temperature
                J_T.row(j) << w * d.x(), w * d.y(), w * d.z();
            }
        }

        /// Density
        qr_inv(J_rho, J_INV_rho.at(c.index - 1));

        /// Velocity
        qr_inv(J_U, J_INV_U.at(c.index - 1));

        /// Pressure
        qr_inv(J_p, J_INV_p.at(c.index - 1));

        /// Temperature
        qr_inv(J_T, J_INV_T.at(c.index - 1));
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

void GRAD_Cell_Density()
{
    for (auto &C : cell)
    {
        const size_t nF = C.surface.size();
        Eigen::VectorXd dr(nF);
        for (size_t i = 0; i < nF; ++i)
        {
            auto f = C.surface.at(i);
            if (f->at_boundary)
            {
                const Scalar w = 1.0 / (f->centroid - C.centroid).norm();
                dr(i) = w * (f->rho - C.rho);
            }
            else
            {
                auto F = C.adjCell.at(i);
                dr(i) = (F->rho - C.rho) / (F->centroid - C.centroid).norm();
            }
        }
        C.grad_rho = J_INV_rho.at(C.index - 1) * dr;
    }
}

void GRAD_Cell_Velocity()
{
    for (auto &C : cell)
    {
        const size_t nF = C.surface.size();
        Eigen::VectorXd du(nF), dv(nF), dw(nF);
        for (size_t i = 0; i < nF; ++i)
        {
            auto f = C.surface.at(i);
            if (f->at_boundary)
            {
                const auto U_BC = f->parent->U_BC;
                if (U_BC == Dirichlet)
                {
                    const Scalar w = 1.0 / (f->centroid - C.centroid).norm();
                    du(i) = w * (f->U.x() - C.U.x());
                    dv(i) = w * (f->U.y() - C.U.y());
                    dw(i) = w * (f->U.z() - C.U.z());
                }
                else if(U_BC == Neumann)
                {
                    du(i) = f->sn_grad_U.x();
                    dv(i) = f->sn_grad_U.y();
                    dw(i) = f->sn_grad_U.z();
                }
                else
                    throw unsupported_boundary_condition(U_BC);
            }
            else
            {
                auto F = C.adjCell.at(i);
                const Scalar w = 1.0 / (F->centroid - C.centroid).norm();
                du(i) = w * (F->U.x() - C.U.x());
                dv(i) = w * (F->U.y() - C.U.y());
                dw(i) = w * (F->U.z() - C.U.z());
            }
        }
        C.grad_U.col(0) = J_INV_U.at(C.index - 1) * du;
        C.grad_U.col(1) = J_INV_U.at(C.index - 1) * dv;
        C.grad_U.col(2) = J_INV_U.at(C.index - 1) * dw;
    }
}

void GRAD_Cell_Pressure()
{
    for (auto &C : cell)
    {
        const size_t nF = C.surface.size();
        Eigen::VectorXd dP(nF);
        for (size_t i = 0; i < nF; ++i)
        {
            auto f = C.surface.at(i);
            if (f->at_boundary)
            {
                const auto p_BC = f->parent->p_BC;
                if (p_BC == Dirichlet)
                {
                    const Scalar w = 1.0 / (f->centroid - C.centroid).norm();
                    dP(i) = w * (f->p - C.p);
                }
                else if (p_BC == Neumann)
                    dP(i) = f->sn_grad_p;
                else
                    throw unsupported_boundary_condition(p_BC);
            }
            else
            {
                auto F = C.adjCell.at(i);
                const Scalar w = 1.0 / (F->centroid - C.centroid).norm();
                dP(i) = w * (F->p - C.p);
            }
        }
        C.grad_p = J_INV_p.at(C.index - 1) * dP;
    }
}

void GRAD_Cell_Temperature()
{
    for (auto &C : cell)
    {
        const size_t nF = C.surface.size();
        Eigen::VectorXd dT(nF);
        for (size_t i = 0; i < nF; ++i)
        {
            auto f = C.surface.at(i);
            if (f->at_boundary)
            {
                const auto T_BC = f->parent->T_BC;
                if (T_BC == Dirichlet)
                {
                    const Scalar w = 1.0 / (f->centroid - C.centroid).norm();
                    dT(i) = w * (f->T - C.T);
                }
                else if (T_BC == Neumann)
                    dT(i) = f->sn_grad_T;
                else
                    throw unsupported_boundary_condition(T_BC);
            }
            else
            {
                auto F = C.adjCell.at(i);
                const Scalar w = 1.0 / (F->centroid - C.centroid).norm();
                dT(i) = w * (F->T - C.T);
            }
        }
        C.grad_T = J_INV_T.at(C.index - 1) * dT;
    }
}

void GRAD_Face_Density()
{
    for (auto &f : face)
    {
        if (f.at_boundary)
        {
            auto c = f.c0 ? f.c0 : f.c1;
            const Vector &n = f.c0 ? f.n01 : f.n10;
            const Vector d = f.c0 ? f.r0 : f.r1;
            const Vector alpha = n / n.dot(d);
            const Tensor beta = Tensor::Identity() - alpha * d.transpose();

            f.grad_rho = alpha * (f.rho - c->rho) + beta * c->grad_rho;
        }
        else
        {
            const Vector d01 = f.c1->centroid - f.c0->centroid;
            const Vector alpha = f.n01 / f.n01.dot(d01);
            const Tensor beta = Tensor::Identity() - alpha * d01.transpose();

            f.grad_rho = alpha * (f.c1->rho - f.c0->rho) + beta * (f.ksi0 * f.c0->grad_rho + f.ksi1 * f.c1->grad_rho);
        }
    }
}

void GRAD_Face_Velocity()
{
    Vector gv[3];
    for (auto &f : face)
    {
        if (f.at_boundary)
        {
            auto c = f.c0 ? f.c0 : f.c1;
            const Vector &n = f.c0 ? f.n01 : f.n10;
            const Vector d = f.c0 ? f.r0 : f.r1;
            const auto U_BC = f.parent->U_BC;
            if (U_BC == Dirichlet)
            {
                const Vector alpha = n / n.dot(d);
                const Tensor beta = Tensor::Identity() - alpha * d.transpose();
                gv[0] = alpha * (f.U.x() - c->U.x()) + beta * c->grad_U.col(0);
                gv[1] = alpha * (f.U.y() - c->U.y()) + beta * c->grad_U.col(1);
                gv[2] = alpha * (f.U.z() - c->U.z()) + beta * c->grad_U.col(2);
            }
            else if (U_BC == Neumann)
            {
                const Tensor gamma = Tensor::Identity() - n * n.transpose();
                gv[0] = n * f.sn_grad_U.x() + gamma * c->grad_U.col(0);
                gv[1] = n * f.sn_grad_U.y() + gamma * c->grad_U.col(1);
                gv[2] = n * f.sn_grad_U.z() + gamma * c->grad_U.col(2);
            }
            else
                throw unsupported_boundary_condition(U_BC);
        }
        else
        {
            const Vector d01 = f.c1->centroid - f.c0->centroid;
            const Vector alpha = f.n01 / f.n01.dot(d01);
            const Tensor beta = Tensor::Identity() - alpha * d01.transpose();

            gv[0] = alpha * (f.c1->U.x() - f.c0->U.x()) + beta * (f.ksi0 * f.c0->grad_U.col(0) + f.ksi1 * f.c1->grad_U.col(0));
            gv[1] = alpha * (f.c1->U.y() - f.c0->U.y()) + beta * (f.ksi0 * f.c0->grad_U.col(1) + f.ksi1 * f.c1->grad_U.col(1));
            gv[2] = alpha * (f.c1->U.z() - f.c0->U.z()) + beta * (f.ksi0 * f.c0->grad_U.col(2) + f.ksi1 * f.c1->grad_U.col(2));
        }
        f.grad_U.col(0) = gv[0];
        f.grad_U.col(1) = gv[1];
        f.grad_U.col(2) = gv[2];
    }
}

void GRAD_Face_Pressure()
{
    for (auto &f : face)
    {
        if (f.at_boundary)
        {
            auto c = f.c0 ? f.c0 : f.c1;
            const Vector &n = f.c0 ? f.n01 : f.n10;
            const Vector d = f.c0 ? f.r0 : f.r1;

            const auto p_BC = f.parent->p_BC;
            if (p_BC == Dirichlet)
            {
                const Vector alpha = n / n.dot(d);
                const Tensor beta = Tensor::Identity() - alpha * d.transpose();
                f.grad_p = alpha * (f.p - c->p) + beta * c->grad_p;
            }
            else if (p_BC == Neumann)
            {
                const Tensor gamma = Tensor::Identity() - n * n.transpose();
                f.grad_p = n * f.sn_grad_p + gamma * c->grad_p;
            }
            else
                throw unsupported_boundary_condition(p_BC);
        }
        else
        {
            const Vector d01 = f.c1->centroid - f.c0->centroid;
            const Vector alpha = f.n01 / f.n01.dot(d01);
            const Tensor beta = Tensor::Identity() - alpha * d01.transpose();
            f.grad_p = alpha * (f.c1->p - f.c0->p) + beta * (f.ksi0 * f.c0->grad_p + f.ksi1 * f.c1->grad_p);
        }
    }
}

void GRAD_Face_Temperature()
{
    for (auto &f : face)
    {
        if (f.at_boundary)
        {
            auto c = f.c0 ? f.c0 : f.c1;
            const Vector &n = f.c0 ? f.n01 : f.n10;
            const Vector d = f.c0 ? f.r0 : f.r1;

            const auto T_BC = f.parent->T_BC;
            if (T_BC == Dirichlet)
            {
                const Vector alpha = n / n.dot(d);
                const Tensor beta = Tensor::Identity() - alpha * d.transpose();
                f.grad_T = alpha * (f.T - c->T) + beta * c->grad_T;
            }
            else if (T_BC == Neumann)
            {
                const Tensor gamma = Tensor::Identity() - n * n.transpose();
                f.grad_T = n * f.sn_grad_T + gamma * c->grad_T;
            }
            else
                throw unsupported_boundary_condition(T_BC);
        }
        else
        {
            const Vector d01 = f.c1->centroid - f.c0->centroid;
            const Vector alpha = f.n01 / f.n01.dot(d01);
            const Tensor beta = Tensor::Identity() - alpha * d01.transpose();
            f.grad_T = alpha * (f.c1->T - f.c0->T)  + beta * (f.ksi0 * f.c0->grad_T + f.ksi1 * f.c1->grad_T);
        }
    }
}

void GRAD_Face_Temperature_next()
{
    for (auto &f : face)
    {
        if (f.at_boundary)
        {
            auto c = f.c0 ? f.c0 : f.c1;
            const Vector &n = f.c0 ? f.n01 : f.n10;
            const Vector d = f.c0 ? f.r0 : f.r1;

            const auto T_BC = f.parent->T_BC;
            if (T_BC == Dirichlet)
            {
                const Vector alpha = n / n.dot(d);
                const Tensor beta = Tensor::Identity() - alpha * d.transpose();
                f.grad_T_next = alpha * (f.T - c->T_next) + beta * c->grad_T_next;
            }
            else if (T_BC == Neumann)
            {
                const Tensor gamma = Tensor::Identity() - n * n.transpose();
                f.grad_T_next = n * f.sn_grad_T + gamma * c->grad_T_next;
            }
            else
                throw unsupported_boundary_condition(T_BC);
        }
        else
        {
            const Vector d01 = f.c1->centroid - f.c0->centroid;
            const Vector alpha = f.n01 / f.n01.dot(d01);
            const Tensor beta = Tensor::Identity() - alpha * d01.transpose();
            f.grad_T_next = alpha * (f.c1->T_next - f.c0->T_next)  + beta * (f.ksi0 * f.c0->grad_T_next + f.ksi1 * f.c1->grad_T_next);
        }
    }
}

void GRAD_Cell_Temperature_star()
{
    for (auto &C : cell)
    {
        const size_t nF = C.surface.size();
        Eigen::VectorXd dT(nF);
        for (size_t i = 0; i < nF; ++i)
        {
            auto f = C.surface.at(i);
            if (f->at_boundary)
            {
                const auto T_BC = f->parent->T_BC;
                if (T_BC == Dirichlet)
                {
                    const Scalar w = 1.0 / (f->centroid - C.centroid).norm();
                    dT(i) = w * (f->T - C.T_star);
                }
                else if (T_BC == Neumann)
                    dT(i) = f->sn_grad_T;
                else
                    throw unsupported_boundary_condition(T_BC);
            }
            else
            {
                auto F = C.adjCell.at(i);
                const Scalar w = 1.0 / (F->centroid - C.centroid).norm();
                dT(i) = w * (F->T_star - C.T_star);
            }
        }
        C.grad_T_star = J_INV_T.at(C.index - 1) * dT;
    }
}

void GRAD_Cell_Temperature_next()
{
    for (auto &C : cell)
    {
        const size_t nF = C.surface.size();
        Eigen::VectorXd dT(nF);
        for (size_t i = 0; i < nF; ++i)
        {
            auto f = C.surface.at(i);
            if (f->at_boundary)
            {
                const auto T_BC = f->parent->T_BC;
                if (T_BC == Dirichlet)
                {
                    const Scalar w = 1.0 / (f->centroid - C.centroid).norm();
                    dT(i) = w * (f->T - C.T_next);
                }
                else if (T_BC == Neumann)
                    dT(i) = f->sn_grad_T;
                else
                    throw unsupported_boundary_condition(T_BC);
            }
            else
            {
                auto F = C.adjCell.at(i);
                const Scalar w = 1.0 / (F->centroid - C.centroid).norm();
                dT(i) = w * (F->T_next - C.T_next);
            }
        }
        C.grad_T_next = J_INV_T.at(C.index - 1) * dT;
    }
}

Scalar GRAD_Cell_PressureCorrection()
{
    Scalar error = 0.0;
    for (auto &c : cell)
    {
        const auto nF = c.surface.size();
        Eigen::VectorXd delta_phi(nF);
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
                default:
                    throw unsupported_boundary_condition(curFace->parent->p_BC);
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

void GRAD_Face_PressureCorrection()
{
    for (auto &f : face)
    {
        if (f.at_boundary)
        {
            auto c = f.c0? f.c0 : f.c1;
            if(f.parent->p_BC == Dirichlet)
            {
                const Vector d = f.centroid - c->centroid;
                /// Zero-Value is assumed.
                f.grad_p_prime = (0.0 - c->p_prime) / d.dot(d) * d;
            }
            else if(f.parent->p_BC == Neumann)
            {
                const Vector &n = f.c0 ? f.n01 : f.n10;
                /// Zero-Gradient is assumed.
                f.grad_p_prime =  (Tensor::Identity() - n * n.transpose()) * c->grad_p_prime;
            }
            else
                throw unsupported_boundary_condition(f.parent->p_BC);
        }
        else
        {
            f.grad_p_prime = f.ksi0 * f.c0->grad_p_prime + f.ksi1 * f.c1->grad_p_prime; /// CDS
        }
    }
}

void GRAD_Cell_Velocity_next()
{
    for (auto &C : cell)
    {
        const size_t nF = C.surface.size();
        Eigen::VectorXd du(nF), dv(nF), dw(nF);
        for (size_t i = 0; i < nF; ++i)
        {
            auto f = C.surface.at(i);
            if (f->at_boundary)
            {
                const auto U_BC = f->parent->U_BC;
                if (U_BC == Dirichlet)
                {
                    const Scalar w = 1.0 / (f->centroid - C.centroid).norm();
                    du(i) = w * (f->U.x() - C.U_next.x());
                    dv(i) = w * (f->U.y() - C.U_next.y());
                    dw(i) = w * (f->U.z() - C.U_next.z());
                }
                else if(U_BC == Neumann)
                {
                    du(i) = f->sn_grad_U.x();
                    dv(i) = f->sn_grad_U.y();
                    dw(i) = f->sn_grad_U.z();
                }
                else
                    throw unsupported_boundary_condition(U_BC);
            }
            else
            {
                auto F = C.adjCell.at(i);
                const Scalar w = 1.0 / (F->centroid - C.centroid).norm();
                du(i) = w * (F->U_next.x() - C.U_next.x());
                dv(i) = w * (F->U_next.y() - C.U_next.y());
                dw(i) = w * (F->U_next.z() - C.U_next.z());
            }
        }
        C.grad_U_next.col(0) = J_INV_U.at(C.index - 1) * du;
        C.grad_U_next.col(1) = J_INV_U.at(C.index - 1) * dv;
        C.grad_U_next.col(2) = J_INV_U.at(C.index - 1) * dw;
    }
}

void GRAD_Face_Velocity_next()
{
    Vector gv[3];
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

            const auto U_BC = f.parent->U_BC;
            if (U_BC == Dirichlet)
            {
                gv[0] = alpha * (f.U.x() - c->U_next.x()) + beta * c->grad_U_next.col(0);
                gv[1] = alpha * (f.U.y() - c->U_next.y()) + beta * c->grad_U_next.col(1);
                gv[2] = alpha * (f.U.z() - c->U_next.z()) + beta * c->grad_U_next.col(2);
            }
            else if (U_BC == Neumann)
            {
                gv[0] = n * f.sn_grad_U.x() + gamma * c->grad_U_next.col(0);
                gv[1] = n * f.sn_grad_U.y() + gamma * c->grad_U_next.col(1);
                gv[2] = n * f.sn_grad_U.z() + gamma * c->grad_U_next.col(2);
            }
            else
                throw unsupported_boundary_condition(U_BC);
        }
        else
        {
            const Vector d01 = f.c1->centroid - f.c0->centroid;
            const Vector alpha = f.n01 / f.n01.dot(d01);
            const Tensor beta = Tensor::Identity() - alpha * d01.transpose();

            gv[0] = alpha * (f.c1->U_next.x() - f.c0->U_next.x()) + beta * (f.ksi0 * f.c0->grad_U_next.col(0) + f.ksi1 * f.c1->grad_U_next.col(0));
            gv[1] = alpha * (f.c1->U_next.y() - f.c0->U_next.y()) + beta * (f.ksi0 * f.c0->grad_U_next.col(1) + f.ksi1 * f.c1->grad_U_next.col(1));
            gv[2] = alpha * (f.c1->U_next.z() - f.c0->U_next.z()) + beta * (f.ksi0 * f.c0->grad_U_next.col(2) + f.ksi1 * f.c1->grad_U_next.col(2));
        }
        f.grad_U_next.col(0) = gv[0];
        f.grad_U_next.col(1) = gv[1];
        f.grad_U_next.col(2) = gv[2];
    }
}

void GRAD_Cell_Pressure_next()
{
    for (auto &C : cell)
    {
        const size_t nF = C.surface.size();
        Eigen::VectorXd dP(nF);
        for (size_t i = 0; i < nF; ++i)
        {
            auto f = C.surface.at(i);
            if (f->at_boundary)
            {
                const auto p_BC = f->parent->p_BC;
                if (p_BC == Dirichlet)
                {
                    const Scalar w = 1.0 / (f->centroid - C.centroid).norm();
                    dP(i) = w * (f->p - C.p_next);
                }
                else if (p_BC == Neumann)
                    dP(i) = f->sn_grad_p;
                else
                    throw unsupported_boundary_condition(p_BC);
            }
            else
            {
                auto F = C.adjCell.at(i);
                const Scalar w = 1.0 / (F->centroid - C.centroid).norm();
                dP(i) = w * (F->p_next - C.p_next);
            }
        }
        C.grad_p_next = J_INV_p.at(C.index - 1) * dP;
    }
}
