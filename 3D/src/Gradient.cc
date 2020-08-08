#include "../inc/custom_type.h"
#include "../inc/Gradient.h"

extern int NumOfPnt;
extern int NumOfFace;
extern int NumOfCell;
extern NaturalArray<Point> pnt;
extern NaturalArray<Face> face;
extern NaturalArray<Cell> cell;
extern NaturalArray<Patch> patch;

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
                switch(curFace->parent->rho_BC)
                {
                case Dirichlet:
                    dphi(i) = curFace->rho - c.rho;
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
                dphi(i) = curAdjCell->rho - c.rho;
            }
        }
        c.grad_rho = c.J_INV_rho * dphi;

        /// velocity-x
        for (int i = 0; i < nF; ++i)
        {
            auto curFace = c.surface.at(i);
            if (curFace->at_boundary)
            {
                switch(curFace->parent->U_BC[0])
                {
                case Dirichlet:
                    dphi(i) = curFace->U.x() - c.U.x();
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
                dphi(i) = curAdjCell->U.x() - c.U.x();
            }
        }
        c.grad_U.col(0) = c.J_INV_U[0] * dphi;

        /// velocity-y
        for (int i = 0; i < nF; ++i)
        {
            auto curFace = c.surface.at(i);
            if (curFace->at_boundary)
            {
                switch (curFace->parent->U_BC[1])
                {
                case Dirichlet:
                    dphi(i) = curFace->U.y() - c.U.y();
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
                dphi(i) = curAdjCell->U.y() - c.U.y();
            }
        }
        c.grad_U.col(1) = c.J_INV_U[1] * dphi;

        /// velocity-z
        for (int i = 0; i < nF; ++i)
        {
            auto curFace = c.surface.at(i);
            if (curFace->at_boundary)
            {
                switch (curFace->parent->U_BC[2])
                {
                case Dirichlet:
                    dphi(i) = curFace->U.z() - c.U.z();
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
                dphi(i) = curAdjCell->U.z() - c.U.z();
            }
        }
        c.grad_U.col(2) = c.J_INV_U[2] * dphi;

        /// pressure
        for (int i = 0; i < nF; ++i)
        {
            auto curFace = c.surface.at(i);
            if (curFace->at_boundary)
            {
                switch (curFace->parent->p_BC)
                {
                case Dirichlet:
                    dphi(i) = curFace->p - c.p;
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
                dphi(i) = curAdjCell->p - c.p;
            }
        }
        c.grad_p = c.J_INV_p * dphi;

        /// temperature
        for (int i = 0; i < nF; ++i)
        {
            auto curFace = c.surface.at(i);
            if (curFace->at_boundary)
            {
                switch (curFace->parent->T_BC)
                {
                case Dirichlet:
                    dphi(i) = curFace->T - c.T;
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
                dphi(i) = curAdjCell->T - c.T;
            }
        }
        c.grad_T = c.J_INV_T * dphi;
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
                    dphi(i) = -c.p_prime; /// Zero-Value is assumed.
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
                dphi(i) = curAdjCell->p_prime - c.p_prime;
            }
        }
        c.grad_p_prime = c.J_INV_p_prime * dphi;
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
static inline void interpGradientToFace(const Vector &gpf0,Scalar sn_gpf,const Vector &n,Vector &dst)
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
            Cell *c = f.c0 ? f.c0 : f.c1;
            bool adj_to_0 = c == f.c0;

            const Vector &r_C = c->centroid;
            const Vector &r_F = f.centroid;
            Vector e_CF = r_F - r_C;
            const Scalar d_CF = e_CF.norm(); /// Distance between cell centroid and face centroid.
            e_CF /= d_CF; /// Unit displacement vector from cell centroid to face centroid.
            const Vector &n = adj_to_0 ? f.n01 : f.n10; /// Unit outward surface normal vector.

            switch (f.parent->p_prime_BC)
            {
            case Dirichlet:
                interpGradientToFace(c->grad_p_prime, c->p_prime, ZERO_SCALAR, e_CF, d_CF, f.grad_p_prime); /// 0-value is assumed.
                break;
            case Neumann:
                interpGradientToFace(c->grad_p_prime, ZERO_SCALAR, n, f.grad_p_prime); /// 0-gradient in normal direction is assumed.
                break;
            case Robin:
                throw robin_bc_is_not_supported();
            }
        }
        else
        {
            const Vector &r_C = f.c0->centroid;
            const Vector &r_F = f.c1->centroid;
            Vector e_CF = r_F - r_C;
            const Scalar d_CF = e_CF.norm();
            e_CF /= d_CF;

            const Vector grad_p_prime_bar = f.ksi0 * f.c0->grad_p_prime + f.ksi1 * f.c1->grad_p_prime;
            const Scalar p_prime_C = f.c0->p_prime;
            const Scalar p_prime_F = f.c1->p_prime;
            interpGradientToFace(grad_p_prime_bar, p_prime_C, p_prime_F, e_CF, d_CF, f.grad_p_prime);
        }
    }
}
