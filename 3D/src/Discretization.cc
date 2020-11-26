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
extern SX_VEC A_dp_2_diag;
extern SX_VEC A_dp_2_diag_unsteady;
extern SX_AMG dp_solver_2;
extern std::string SEP;
extern std::ostream& LOG_OUT;

static const Scalar Rg = 287.7; // J / (Kg * K)

static void calcBoundaryFacePrimitiveValue(Face& f, Cell* c, const Vector& d)
{
    auto p = f.parent;

    if (p->rho_BC == Neumann)
        f.rho = c->rho + f.grad_rho.dot(d);

    if (p->U_BC[0] == Neumann)
        f.U.x() = c->U.x() + f.grad_U.col(0).dot(d);

    if (p->U_BC[1] == Neumann)
        f.U.y() = c->U.y() + f.grad_U.col(1).dot(d);

    if (p->U_BC[2] == Neumann)
        f.U.z() = c->U.z() + f.grad_U.col(2).dot(d);

    if (p->p_BC == Neumann)
        f.p = c->p + f.grad_p.dot(d);

    if (p->T_BC == Neumann)
        f.T = c->T + f.grad_T.dot(d);
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
                const Vector tw = -f.viscosity / r.dot(n) * dU;
                f.tau = tw * n.transpose();
            }
            else if (p->BC == BC_PHY::Symmetry)
            {
                Vector dU = c->U.dot(n) * n;
                const Vector t_cz = -2.0 * f.viscosity * dU / r.norm();
                f.tau = t_cz * n.transpose();
            }
            else if (p->BC == BC_PHY::Inlet || p->BC == BC_PHY::Outlet)
                Stokes(f.viscosity, f.grad_U, f.tau);
            else
                throw unsupported_boundary_condition(p->BC);
        }
        else
            Stokes(f.viscosity, f.grad_U, f.tau);
    }
}

void calc_face_viscous_shear_stress_next()
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
                Vector dU = c->U_next - f.U_next;
                dU -= dU.dot(n) * n;
                const Vector tw = -f.viscosity / r.dot(n) * dU;
                f.tau_next = tw * n.transpose();
            }
            else if (p->BC == BC_PHY::Symmetry)
            {
                Vector dU = c->U_next.dot(n) * n;
                const Vector t_cz = -2.0 * f.viscosity * dU / r.norm();
                f.tau_next = t_cz * n.transpose();
            }
            else if (p->BC == BC_PHY::Inlet || p->BC == BC_PHY::Outlet)
                Stokes(f.viscosity, f.grad_U_next, f.tau_next);
            else
                throw unsupported_boundary_condition(p->BC);
        }
        else
            Stokes(f.viscosity, f.grad_U_next, f.tau_next);
    }
}

static int pcpe(Scalar TimeStep)
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
        update_rhs(&Q_dp_2, TimeStep);
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

Scalar double_dot(const Tensor &A, const Tensor &B)
{
    Scalar ret = 0.0;
    for(int i = 0; i < 3; ++i)
        for(int j = 0; j < 3; ++j)
            ret += A(i, j) * B(j, i);
    return ret;
}

void ForwardEuler(Scalar TimeStep)
{
    /// Auxiliary vars @(n)
    for (auto& c : cell)
    {
        c.viscosity = c.rho / Re;
        c.specific_heat_p = 3.5 * Rg;
        c.specific_heat_v = c.specific_heat_p / 1.4;
        c.conductivity = c.specific_heat_p * c.viscosity / 0.72;
    }
    set_bc_of_primitive_var();
    calc_cell_primitive_gradient();
    calc_face_primitive_gradient();
    calc_face_primitive_var();
    for (auto& f : face)
    {
        f.viscosity = f.rho / Re;
        f.specific_heat_p = 3.5 * Rg;
        f.specific_heat_v = f.specific_heat_v / 1.4;
        f.conductivity = f.specific_heat_p * f.viscosity / 0.72;
    }
    calc_face_viscous_shear_stress();
    set_bc_of_conservative_var();
    set_bc_of_pressure_correction();

    /// Init @(m-1)
    for (auto& c : cell)
    {
        c.rho_prev = c.rho;
        c.p_prev = c.p;
        c.T_prev = c.T;
    }
    for (auto& f : face)
    {
        f.p_prev = f.p;
    }

    /// Loop @(m)
    int m = 0;
    while(++m < 3)
    {
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
                convection_flux += (f->rhoU.dot(Sf) * f->U);
                pressure_flux += (f->p_prev * Sf);
                viscous_flux += (f->tau * Sf);
            }
            c.rhoU_star = c.rhoU + TimeStep / c.volume * (-convection_flux - pressure_flux + viscous_flux);
            c.U_star = c.rhoU_star / c.rho_prev;
        }

        /// rhoU* on each face
        for (auto& f : face)
        {
            if (f.at_boundary)
                f.rhoU_star = f.rhoU;
            else
            {
                f.rhoU_star = f.ksi0 * f.c0->rhoU_star + f.ksi1 * f.c1->rhoU_star;

                /// Rhie-Chow interpolation
                const Vector mean_grad_p = 0.5 * (f.c1->grad_p + f.c0->grad_p);
                const Vector d = f.c1->centroid - f.c0->centroid;
                const Vector compact_grad_p = (f.c1->p - f.c0->p) / (d.dot(d)) * d;
                const Vector rhoU_rc = -TimeStep * (compact_grad_p - mean_grad_p);
                f.rhoU_star += rhoU_rc;
            }
            f.U_star = f.rhoU_star / f.rho_prev;
        }

        /// Continuity imbalance
        for (auto& c : cell)
        {
            c.dmdt = c.volume / TimeStep * (c.rho_prev - c.rho);
            for (size_t j = 0; j < c.surface.size(); ++j)
            {
                auto f = c.surface.at(j);
                const auto& Sf = c.S.at(j);
                c.dmdt += f->rhoU_star.dot(Sf);
            }
        }

        /// $\frac{\partial \rho}{\partial p}$
        for(auto &c : cell)
        {
            c.drhodp_prev = 1.0 / (Rg * c.T_prev);
        }

        /// Contribution to the diagonal of Poisson equation
        for(auto &c : cell)
        {
            const size_t idx = c.index - 1;
            const Scalar val = c.volume * c.drhodp_prev / std::pow(TimeStep, 2);
            sx_vec_set_entry(&A_dp_2_diag_unsteady, idx, val);
        }

        /// Update coefficient of Poisson equation
        update_diag(&A_dp_2, &A_dp_2_diag, &A_dp_2_diag_unsteady);

        /// Correction Step
        LOG_OUT << "\n" << SEP << "Solving pressure-correction ..." << std::endl;
        LOG_OUT << SEP << "--------------------------------------------" << std::endl;
        LOG_OUT << SEP << "||p'-p'_prev||    ||grad(p')-grad(p')_prev||" << std::endl;
        LOG_OUT << SEP << "--------------------------------------------" << std::endl;
        const int poisson_noc_iter = pcpe(TimeStep);
        LOG_OUT << SEP << "--------------------------------------------" << std::endl;
        LOG_OUT << SEP << "Converged after " << poisson_noc_iter << " iterations" << std::endl;

        /// Calculate $\frac{\partial p'}{\partial n}$ on face centroid
        for (auto &f : face)
        {
            if (f.at_boundary)
            {
                const Vector &n = f.c0 ? f.n01 : f.n10;
                f.sn_grad_p_prime = (f.grad_p_prime.dot(n)) * n;
            }
            else
            {
                const Vector d01 = f.c1->centroid - f.c0->centroid;
                const Vector e01 = d01 / d01.norm();
                const Scalar alpha = 1.0/f.n01.dot(e01);
                const Scalar tmp1 = alpha * (f.c1->p_prime - f.c0->p_prime) / d01.norm();
                const Scalar tmp2 = f.grad_p_prime.dot(f.n01 - alpha * e01);
                f.sn_grad_p_prime = (tmp1 + tmp2) * f.n01;
            }
        }

        /// Reconstruct gradient of $p'$ at cell centroid
        for (auto& c : cell)
        {
            Vector b(0.0, 0.0, 0.0);
            const size_t Nf = c.surface.size();
            for (size_t j = 0; j < Nf; ++j)
            {
                auto f = c.surface.at(j);
                const Vector &Sf = c.S.at(j);
                b += f->sn_grad_p_prime.dot(Sf) * Sf / f->area;
            }
            c.grad_p_prime = c.TeC_INV * b;
        }

        /// Update
        for (auto& f : face)
        {
            if (!f.at_boundary)
            {
                f.rho_prime = f.drhodp_prev * f.p_prime;
                f.rho_star = f.rho_prev + f.rho_prime;
                f.U_prime = - TimeStep * f.sn_grad_p_prime / f.rho_prev;
                f.U_next = f.U_star + f.U_prime;
                f.rhoU_next = f.rhoU_star + f.rho_prev * f.U_prime + f.rho_prime * f.U_star;
                f.p_next = f.p_prev + f.p_prime;
            }
        }
        for (auto& c : cell)
        {
            c.rho_prime = c.drhodp_prev * c.p_prime;
            c.rho_star = c.rho_prev + c.rho_prime;
            c.U_prime = - TimeStep * c.grad_p_prime / c.rho_prev;
            c.U_next = c.U_star + c.U_prime;
            c.rhoU_next = c.rhoU_star + c.rho_prev * c.U_prime + c.rho_prime * c.U_star;
            c.p_next = c.p_prev + c.p_prime;
        }

        /// Gradient of these updated quantities
        calc_cell_primitive_gradient_next();
        calc_face_viscous_shear_stress_next();

        /// Prediction of enthalpy
        for(auto &c : cell)
        {
            Scalar convection_flux = 0.0;
            Scalar diffusion_flux = 0.0;
            const auto Nf = c.S.size();
            for (int j = 0; j < Nf; ++j)
            {
                auto f = c.surface.at(j);
                const auto &Sf = c.S.at(j);

                convection_flux += f->rhoU_next.dot(Sf) * f->h;
                diffusion_flux += f->conductivity * f->grad_T.dot(Sf);
            }
            const Scalar pressure_work = c.U_next.dot(c.grad_p_next) * c.volume;
            const Scalar viscous_dissipation = double_dot(c.tau_next, c.grad_U_next) * c.volume;
            const Scalar dpdt = (c.p_next - c.p) / TimeStep * c.volume;
            c.rhoh_next = c.rhoh + TimeStep / c.volume * (-convection_flux + diffusion_flux + viscous_dissipation + pressure_work + dpdt);
            c.h_next = c.rhoh_next / c.rho_star;
            c.T_next = c.h_next / c.specific_heat_p;
            c.rho_next = c.p_next / (Rg * c.T_next);
        }
    }

    for(auto &c : cell)
    {
        c.rho = c.rho_next;
        c.U = c.U_next;
        c.p = c.p_next;
        c.T = c.T_next;
        c.h = c.h_next;
        c.rhoU = c.rhoU_next;
        c.rhoh = c.rhoh_next;
    }
}
