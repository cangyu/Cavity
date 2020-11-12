#include <iostream>
#include <iomanip>
#include "../inc/BC.h"
#include "../inc/PoissonEqn.h"
#include "../inc/CHEM.h"
#include "../inc/Gradient.h"
#include "../inc/Discretization.h"

extern int NumOfPnt, NumOfFace, NumOfCell;
extern NaturalArray<Point> pnt;
extern NaturalArray<Face> face;
extern NaturalArray<Cell> cell;
extern NaturalArray<Patch> patch;
extern Scalar Re;
extern SX_MAT A_dp_2;
extern SX_VEC Q_dp_2;
extern SX_VEC x_dp_2;
extern SX_AMG dp_solver_2;
extern std::string SEP;
extern std::ostream& LOG_OUT;

static void calcBoundaryFacePrimitiveValue(Face& f, Cell* c, const Vector& d)
{
    auto p = f.parent;

    /// density
    switch (p->rho_BC)
    {
    case Dirichlet:
        break;
    case Neumann:
        f.rho = c->rho + f.grad_rho.dot(d);
        break;
    case Robin:
        throw robin_bc_is_not_supported();
    default:
        break;
    }

    /// velocity-x
    switch (p->U_BC[0])
    {
    case Dirichlet:
        break;
    case Neumann:
        f.U.x() = c->U.x() + f.grad_U.col(0).dot(d);
        break;
    case Robin:
        throw robin_bc_is_not_supported();
    default:
        break;
    }

    /// velocity-y
    switch (p->U_BC[1])
    {
    case Dirichlet:
        break;
    case Neumann:
        f.U.y() = c->U.y() + f.grad_U.col(1).dot(d);
        break;
    case Robin:
        throw robin_bc_is_not_supported();
    default:
        break;
    }

    /// velocity-z
    switch (p->U_BC[2])
    {
    case Dirichlet:
        break;
    case Neumann:
        f.U.z() = c->U.z() + f.grad_U.col(2).dot(d);
        break;
    case Robin:
        throw robin_bc_is_not_supported();
    default:
        break;
    }

    /// pressure
    switch (p->p_BC)
    {
    case Dirichlet:
        break;
    case Neumann:
        f.p = c->p + f.grad_p.dot(d);
        break;
    case Robin:
        throw robin_bc_is_not_supported();
    default:
        break;
    }

    /// temperature
    switch (p->T_BC)
    {
    case Dirichlet:
        break;
    case Neumann:
        f.T = c->T + f.grad_T.dot(d);
        break;
    case Robin:
        throw robin_bc_is_not_supported();
    default:
        break;
    }
}

static void calcInternalFacePrimitiveValue(Face& f)
{
    /// pressure
    const Scalar p_0 = f.c0->p + f.c0->grad_p.dot(f.r0);
    const Scalar p_1 = f.c1->p + f.c1->grad_p.dot(f.r1);
    f.p = 0.5 * (p_0 + p_1);

    /// temperature
    const Scalar T_0 = f.c0->T + f.c0->grad_T.dot(f.r0);
    const Scalar T_1 = f.c1->T + f.c1->grad_T.dot(f.r1);
    f.T = f.ksi0 * T_0 + f.ksi1 * T_1;

    /// velocity
    if (f.rhoU.dot(f.n01) > 0)
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

    /// density
    if (f.rhoU.dot(f.n01) > 0)
        f.rho = f.c0->rho + f.c0->grad_rho.dot(f.r0);
    else
        f.rho = f.c1->rho + f.c1->grad_rho.dot(f.r1);
}

void calc_face_primitive_var()
{
    for (auto& f : face)
    {
        if (f.at_boundary)
        {
            if (f.c0)
                calcBoundaryFacePrimitiveValue(f, f.c0, f.r0);
            else if (f.c1)
                calcBoundaryFacePrimitiveValue(f, f.c1, f.r1);
            else
                throw empty_connectivity(f.index);
        }
        else
            calcInternalFacePrimitiveValue(f);
    }
}

void calc_face_viscous_shear_stress()
{
    for (auto& f : face)
    {
        if (f.at_boundary)
        {
            auto c = f.c0 ? f.c0 : f.c1;
            const Vector& n = f.c0 ? f.n01 : f.n10;
            const Vector& r = f.c0 ? f.r0 : f.r1;

            auto p = f.parent;
            if (p->BC == BC_PHY::Wall)
            {
                Vector dU = c->U - f.U;
                dU -= dU.dot(n) * n;
                const Vector tw = -f.mu / r.dot(n) * dU;
                f.tau = tw * n.transpose();
            }
            else if (p->BC == BC_PHY::Symmetry)
            {
                Vector dU = c->U.dot(n) * n;
                const Vector t_cz = -2.0 * f.mu * dU / r.norm();
                f.tau = t_cz * n.transpose();
            }
            else if (p->BC == BC_PHY::Inlet || p->BC == BC_PHY::Outlet)
                Stokes(f.mu, f.grad_U, f.tau);
            else
                throw unsupported_boundary_condition(p->BC);
        }
        else
            Stokes(f.mu, f.grad_U, f.tau);
    }
}

void reconstruction()
{
    /// Update physical properties at centroid of each cell.
    for (auto& c : cell)
    {
        /// Dynamic viscosity
        // c.mu = Sutherland(c.T);
        c.mu = c.rho / Re;
    }

    /// Enforce boundary conditions for primitive variables.
    set_bc_of_primitive_var();

    /// Gradients of primitive variables at centroid of each cell.
    calc_cell_primitive_gradient();

    /// Gradients of primitive variables at centroid of each face.
    calc_face_primitive_gradient();

    /// Interpolate values of primitive variables on each face.
    calc_face_primitive_var();

    /// Update physical properties at centroid of each face.
    for (auto& f : face)
    {
        /// Dynamic viscosity
        // f.mu = Sutherland(f.T);
        f.mu = f.rho / Re;
    }

    /// Viscous shear stress on each face.
    calc_face_viscous_shear_stress();

    set_bc_of_conservative_var();

    set_bc_of_pressure_correction();
}

static int ppe(Scalar TimeStep)
{
    for(auto &c : cell)
    {
        c.p_prime = ZERO_SCALAR;
        c.grad_p_prime.setZero();
    }
    for(auto &f : face)
        f.grad_p_prime.setZero();

    int cnt = 0; /// Iteration counter
    Scalar l1 = 1.0, l2 = 1.0; /// Convergence monitor
    while (l1 > 1e-10 || l2 > 1e-8)
    {
        /// Solve p' at cell centroid
        calcPressureCorrectionEquationRHS(Q_dp_2, TimeStep);
        sx_solver_amg_solve(&dp_solver_2, &x_dp_2, &Q_dp_2);
        l1 = 0.0;
        for (int i = 0; i < NumOfCell; ++i)
        {
            auto& c = cell.at(i);
            const Scalar new_val = sx_vec_get_entry(&x_dp_2, i);
            l1 += std::fabs(new_val - c.p_prime);
            c.p_prime = new_val;
        }
        l1 /= NumOfCell;

        /// Calculate gradient of $p'$ at cell centroid
        l2 = calc_cell_pressure_correction_gradient();

        /// Interpolate gradient of $p'$ from cell centroid to face centroid
        calc_face_pressure_correction_gradient();

        /// Report
        LOG_OUT << SEP << std::left << std::setw(14) << l1 << "    " << std::setw(26) << l2 << std::endl;

        /// Next loop if needed
        ++cnt;
    }
    return cnt;
}

/**
 * 1st-order explicit time-marching.
 * Pressure-Velocity coupling is solved using Fractional-Step Method.
 * @param TimeStep
 */
void ForwardEuler(Scalar TimeStep)
{
    reconstruction();

    /// Prediction of momentum
    for (auto& c : cell)
    {
        Vector pressure_flux(0.0, 0.0, 0.0);
        Vector convection_flux(0.0, 0.0, 0.0);
        Vector viscous_flux(0.0, 0.0, 0.0);
        const auto Nf = c.S.size();
        for (int j = 0; j < Nf; ++j)
        {
            auto f = c.surface.at(j);
            const auto& Sf = c.S.at(j);

            /// convection term
            const Vector cur_convection_flux = f->rhoU.dot(Sf) * f->U;
            convection_flux += cur_convection_flux;

            /// pressure term
            const Vector cur_pressure_flux = f->p * Sf;
            pressure_flux += cur_pressure_flux;

            /// viscous term
            const Vector cur_viscous_flux = f->tau * Sf;
            viscous_flux += cur_viscous_flux;
        }

        c.rhoU_star = c.rhoU + TimeStep / c.volume * (-convection_flux - pressure_flux + viscous_flux);
    }

    /// rhoU* on each face
    for (auto& f : face)
    {
        if (f.at_boundary)
            f.rhoU_star = f.rhoU;
        else
        {
            f.rhoU_star = f.ksi0 * f.c0->rhoU_star + f.ksi1 * f.c1->rhoU_star;

            /// Momentum interpolation
            const Vector d01 = f.c1->centroid - f.c0->centroid;
            const Vector compact_grad_p = (f.c1->p - f.c0->p) / (d01.dot(d01)) * d01;
            const Vector mean_grad_p = 0.5 * (f.c1->grad_p + f.c0->grad_p);
            f.rhoU_star -= TimeStep * (compact_grad_p - mean_grad_p);
        }
    }

    /// Correction Step
    LOG_OUT << "\n" << SEP << "Solving pressure-correction ..." << std::endl;
    LOG_OUT << SEP << "--------------------------------------------" << std::endl;
    LOG_OUT << SEP << "||p'-p'_prev||    ||grad(p')-grad(p')_prev||" << std::endl;
    LOG_OUT << SEP << "--------------------------------------------" << std::endl;
    const int poisson_noc_iter = ppe(TimeStep);
    LOG_OUT << SEP << "--------------------------------------------" << std::endl;
    LOG_OUT << SEP << "Converged after " << poisson_noc_iter << " iterations" << std::endl;

    /// Calculate $\frac{\partial p'}{\partial n}$ on face centroid
    for (auto &f : face)
    {
        if (f.at_boundary)
        {
            const Vector &n = f.c0 ? f.n01 : f.n10;
            f.grad_p_prime_sn = (f.grad_p_prime.dot(n)) * n;
        }
        else
        {
            const Vector d01 = f.c1->centroid - f.c0->centroid;
            const Vector e01 = d01 / d01.norm();
            const Scalar alpha = 1.0/f.n01.dot(e01);
            const Scalar tmp1 = alpha * (f.c1->p_prime - f.c0->p_prime) / d01.norm();
            const Scalar tmp2 = f.grad_p_prime.dot(f.n01 - alpha * e01);
            f.grad_p_prime_sn = (tmp1 + tmp2) * f.n01;
        }
    }

    /// Update
    for (auto& f : face)
    {
        if (!f.at_boundary)
            f.rhoU = f.rhoU_star - TimeStep * f.grad_p_prime_sn;
    }

    for (auto& c : cell)
    {
        /// Reconstruct gradient of $p'$ at cell centroid
        Vector b;
        b.setZero();
        const size_t Nf = c.surface.size();
        for(size_t j = 0; j < Nf; ++j)
        {
            auto f = c.surface.at(j);
            const Vector &Sf = c.S.at(j);
            b += f->grad_p_prime_sn.dot(Sf) * Sf / f->area;
        }
        c.grad_p_prime = c.grad_p_prime_rm * b;

        /// Velocity
        c.rhoU = c.rhoU_star - TimeStep * c.grad_p_prime;
        c.U = c.rhoU / c.rho;

        /// Pressure
        c.p += c.p_prime;
    }
}
