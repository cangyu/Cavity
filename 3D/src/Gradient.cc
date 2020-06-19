#include "../inc/custom_type.h"
#include "../inc/Gradient.h"

extern int NumOfPnt, NumOfFace, NumOfCell;
extern NaturalArray<Point> pnt;
extern NaturalArray<Face> face;
extern NaturalArray<Cell> cell;
extern NaturalArray<Patch> patch;

void calcFaceGhostVariable()
{
    for (auto &f : face)
    {
        if (f.atBdry)
        {
            if (f.c0)
            {
                /// density
                switch (f.rho_BC)
                {
                case Neumann:
                    f.rho_ghost = f.c0->rho + 2 * f.sn_grad_rho * f.r0.norm(); /// May not accurate, should be validated or corrected further.
                    break;
                case Dirichlet:
                    f.rho_ghost = f.rho; /// Will not be used.
                    break;
                case Robin:
                    throw robin_bc_is_not_supported();
                default:
                    break;
                }

                /// velocity-x
                switch (f.U_BC[0])
                {
                case Neumann:
                    f.U_ghost.x() = f.c0->U[0] + 2 * f.sn_grad_U.x() * f.r0.norm(); /// May not accurate, should be validated or corrected further.
                    break;
                case Dirichlet:
                    f.U_ghost.x() = f.U.x(); /// Will not be used.
                    break;
                case Robin:
                    throw robin_bc_is_not_supported();
                default:
                    break;
                }

                /// velocity-y
                switch (f.U_BC[1])
                {
                case Neumann:
                    f.U_ghost.y() = f.c0->U[1] + 2 * f.sn_grad_U.y() * f.r0.norm(); /// May not accurate, should be validated or corrected further.
                    break;
                case Dirichlet:
                    f.U_ghost.y() = f.U.y(); /// Will not be used.
                    break;
                case Robin:
                    throw robin_bc_is_not_supported();
                default:
                    break;
                }

                /// velocity-z
                switch (f.U_BC[2])
                {
                case Neumann:
                    f.U_ghost.z() = f.c0->U[2] + 2 * f.sn_grad_U.z() * f.r0.norm(); /// May not accurate, should be validated or corrected further.
                    break;
                case Dirichlet:
                    f.U_ghost.z() = f.U.z(); /// Will not be used.
                    break;
                case Robin:
                    throw robin_bc_is_not_supported();
                default:
                    break;
                }

                /// pressure
                switch (f.p_BC)
                {
                case Neumann:
                    f.p_ghost = f.c0->p + 2 * f.sn_grad_p * f.r0.norm(); /// May not accurate, should be validated or corrected further.
                    break;
                case Dirichlet:
                    f.p_ghost = f.p; /// Will not be used.
                    break;
                case Robin:
                    throw robin_bc_is_not_supported();
                default:
                    break;
                }

                /// temperature
                switch (f.T_BC)
                {
                case Neumann:
                    f.T_ghost = f.c0->T + 2 * f.sn_grad_T * f.r0.norm(); /// May not accurate, should be validated or corrected further.
                    break;
                case Dirichlet:
                    f.T_ghost = f.T; /// Will not be used.
                    break;
                case Robin:
                    throw robin_bc_is_not_supported();
                default:
                    break;
                }
            }
            else if (f.c1)
            {
                /// density
                switch (f.rho_BC)
                {
                case Neumann:
                    f.rho_ghost = f.c1->rho + 2 * f.sn_grad_rho * f.r1.norm();  /// May not accurate, should be validated or corrected further.
                    break;
                case Dirichlet:
                    f.rho_ghost = f.rho; /// Will not be used.
                    break;
                case Robin:
                    throw robin_bc_is_not_supported();
                default:
                    break;
                }

                /// velocity-x
                switch (f.U_BC[0])
                {
                case Neumann:
                    f.U_ghost.x() = f.c1->U[0] + 2 * f.sn_grad_U.x() * f.r1.norm(); /// May not accurate, should be validated or corrected further.
                    break;
                case Dirichlet:
                    f.U_ghost.x() = f.U.x(); /// Will not be used.
                    break;
                case Robin:
                    throw robin_bc_is_not_supported();
                default:
                    break;
                }

                /// velocity-y
                switch (f.U_BC[1])
                {
                case Neumann:
                    f.U_ghost.y() = f.c1->U[1] + 2 * f.sn_grad_U.y() * f.r1.norm(); /// May not accurate, should be validated or corrected further.
                    break;
                case Dirichlet:
                    f.U_ghost.y() = f.U.y(); /// Will not be used.
                    break;
                case Robin:
                    throw robin_bc_is_not_supported();
                default:
                    break;
                }

                /// velocity-z
                switch (f.U_BC[2])
                {
                case Neumann:
                    f.U_ghost.z() = f.c1->U[2] + 2 * f.sn_grad_U.z() * f.r1.norm(); /// May not accurate, should be validated or corrected further.
                    break;
                case Dirichlet:
                    f.U_ghost.z() = f.U.z(); /// Will not be used.
                    break;
                case Robin:
                    throw robin_bc_is_not_supported();
                default:
                    break;
                }

                /// pressure
                switch (f.p_BC)
                {
                case Neumann:
                    f.p_ghost = f.c1->p + 2 * f.sn_grad_p * f.r1.norm(); /// May not accurate, should be validated or corrected further.
                    break;
                case Dirichlet:
                    f.p_ghost = f.p; /// Will not be used.
                    break;
                case Robin:
                    throw robin_bc_is_not_supported();
                default:
                    break;
                }

                /// temperature
                switch (f.T_BC)
                {
                case Neumann:
                    f.T_ghost = f.c1->T + 2 * f.sn_grad_T * f.r1.norm(); /// May not accurate, should be validated or corrected further.
                    break;
                case Dirichlet:
                    f.T_ghost = f.T; /// Will not be used.
                    break;
                case Robin:
                    throw robin_bc_is_not_supported();
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

        /// gradient of density
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

        /// gradient of x-dim velocity
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

        /// gradient of y-dim velocity
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

        /// gradient of z-dim velocity
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

        /// gradient of pressure
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

        /// gradient of temperature
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
            {
                if (curFace->p_prime_BC == Dirichlet)
                    dphi(i) = -c.p_prime; /// Zero-Value is assumed.
                else
                    dphi(i) = 0.0; /// Zero-Gradient is assumed.
            }
            else
                dphi(i) = curAdjCell->p_prime - c.p_prime;
        }
        c.grad_p_prime = c.J_INV_p_prime * dphi;
    }
}

static inline Vector interpGradientToFace(const Vector &predicted_grad_phi_f, Scalar phi_C, Scalar phi_F, const Vector &e_CF, Scalar d_CF)
{
    return predicted_grad_phi_f + ((phi_F - phi_C) / d_CF - predicted_grad_phi_f.dot(e_CF))*e_CF;
}

void calcFaceGradient()
{
    for (auto &f : face)
    {
        if (f.atBdry)
        {
            /// gradients at boundary face
            if (f.c0)
            {
                const Vector &r_C = f.c0->center;
                const Vector &r_F = f.center;
                Vector e_CF = r_F - r_C;
                const Scalar d_CF = e_CF.norm();
                e_CF /= d_CF;

                /// density
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

                /// velocity-x
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

                /// velocity-y
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

                /// velocity-z
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

                /// pressure
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

                /// temperature
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

                /// density
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

                /// velocity-x
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

                /// velocity-y
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

                /// velocity-z
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

                /// pressure
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

                /// temperature
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
            /// gradients at internal face
            const Vector &r_C = f.c0->center;
            const Vector &r_F = f.c1->center;
            Vector e_CF = r_F - r_C;
            const Scalar d_CF = e_CF.norm();
            e_CF /= d_CF;

            /// density
            const Vector predicted_grad_rho = f.ksi0 * f.c0->grad_rho + f.ksi1 * f.c1->grad_rho;
            const Scalar rho_C = f.c0->rho;
            const Scalar rho_F = f.c1->rho;
            f.grad_rho = interpGradientToFace(predicted_grad_rho, rho_C, rho_F, e_CF, d_CF);

            /// velocity-x
            const Vector predicted_grad_u = f.ksi0 * f.c0->grad_U.col(0) + f.ksi1 * f.c1->grad_U.col(0);
            const Scalar u_C = f.c0->U.x();
            const Scalar u_F = f.c1->U.x();
            f.grad_U.col(0) = interpGradientToFace(predicted_grad_u, u_C, u_F, e_CF, d_CF);

            /// velocity-y
            const Vector predicted_grad_v = f.ksi0 * f.c0->grad_U.col(1) + f.ksi1 * f.c1->grad_U.col(1);
            const Scalar v_C = f.c0->U.y();
            const Scalar v_F = f.c1->U.y();
            f.grad_U.col(1) = interpGradientToFace(predicted_grad_v, v_C, v_F, e_CF, d_CF);

            /// velocity-z
            const Vector predicted_grad_w = f.ksi0 * f.c0->grad_U.col(2) + f.ksi1 * f.c1->grad_U.col(2);
            const Scalar w_C = f.c0->U.z();
            const Scalar w_F = f.c1->U.z();
            f.grad_U.col(2) = interpGradientToFace(predicted_grad_w, w_C, w_F, e_CF, d_CF);

            /// pressure
            const Vector predicted_grad_p = f.ksi0 * f.c0->grad_p + f.ksi1 * f.c1->grad_p;
            const Scalar p_C = f.c0->p;
            const Scalar p_F = f.c1->p;
            f.grad_p = interpGradientToFace(predicted_grad_p, p_C, p_F, e_CF, d_CF);

            /// temperature
            const Vector predicted_grad_T = f.ksi0 * f.c0->grad_T + f.ksi1 * f.c1->grad_T;
            const Scalar T_C = f.c0->T;
            const Scalar T_F = f.c1->T;
            f.grad_T = interpGradientToFace(predicted_grad_T, T_C, T_F, e_CF, d_CF);
        }
    }
}

void calcFacePressureCorrectionGradient()
{
    for (auto &f : face)
    {
        if (f.atBdry)
        {
            auto c = f.c0;
            if(!c)
                c = f.c1;

            const Vector &r_C = c->center;
            const Vector &r_F = f.center;
            Vector e_CF = r_F - r_C;
            const Scalar d_CF = e_CF.norm();
            e_CF /= d_CF;

            const Vector grad_p_prime_bar = c->grad_p_prime;

            if(f.p_prime_BC == Dirichlet)
            {
                const Scalar p_prime_C = c->p_prime;
                const Scalar p_prime_F = ZERO_SCALAR;
                f.grad_p_prime = interpGradientToFace(grad_p_prime_bar, p_prime_C, p_prime_F, e_CF, d_CF);
            }
            else if(f.p_prime_BC == Neumann)
            {
                f.grad_p_prime = grad_p_prime_bar - (grad_p_prime_bar.dot(f.n01)) * f.n01;
            }
            else
                throw unsupported_boundary_condition(f.p_prime_BC);
        }
        else
        {
            const Vector &r_C = f.c0->center;
            const Vector &r_F = f.c1->center;
            Vector e_CF = r_F - r_C;
            const Scalar d_CF = e_CF.norm();
            e_CF /= d_CF;

            const Vector grad_p_prime_bar = f.ksi0 * f.c0->grad_p_prime + f.ksi1 * f.c1->grad_p_prime;
            const Scalar p_prime_C = f.c0->p_prime;
            const Scalar p_prime_F = f.c1->p_prime;
            f.grad_p_prime = interpGradientToFace(grad_p_prime_bar, p_prime_C, p_prime_F, e_CF, d_CF);
        }
    }
}
