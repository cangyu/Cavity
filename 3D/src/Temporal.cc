#include <iostream>
#include <iomanip>
#include "../inc/BC.h"
#include "../inc/PoissonEqn.h"
#include "../inc/CHEM.h"
#include "../inc/Gradient.h"
#include "../inc/MISC.h"
#include "../inc/Temporal.h"

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

static void calcBoundaryFacePrimitiveValue(Face& f, Cell* c, const Vector& d)
{
    auto p = f.parent;

    /// velocity
    if(p->U_BC == Neumann)
        f.U = c->U + d.transpose() * f.grad_U;

    /// pressure
    if (p->p_BC == Neumann)
        f.p = c->p + f.grad_p.dot(d);

    /// temperature
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

static void step1(Scalar TimeStep)
{
    for (auto &c : cell)
    {
        Scalar convection_flux = 0.0;
        const auto Nf = c.S.size();
        for (int j = 0; j < Nf; ++j)
        {
            auto f = c.surface.at(j);
            const auto& Sf = c.S.at(j);
            convection_flux += f->rhoU.dot(Sf);
        }
        c.rho_prev = c.rho + TimeStep / c.volume * (-convection_flux);
        c.U_prev = c.U;
        c.p_prev = c.p;
        c.T_prev = c.T;
        c.tau_prev = c.tau;
        c.grad_U_prev = c.grad_U;
        c.grad_p_prev = c.grad_p;
    }
    for (auto &f : face)
    {
        f.rho_prev = f.rho;
        f.U_prev = f.U;
        f.p_prev = f.p;
        f.T_prev = f.T;
        f.h_prev = f.h;
        f.rhoU_prev = f.rhoU;
        f.grad_T_prev = f.grad_T;
        f.tau_prev = f.tau;
    }
}

static void step2(Scalar TimeStep)
{
    /// Prediction of energy
    for(auto &c : cell)
    {
        Scalar convection_flux = 0.0;
        Scalar diffusion_flux = 0.0;
        const auto Nf = c.S.size();
        for (int j = 0; j < Nf; ++j)
        {
            auto f = c.surface.at(j);
            const auto &Sf = c.S.at(j);
            convection_flux += f->rhoU_prev.dot(Sf) * f->h_prev;
            diffusion_flux += f->conductivity * f->grad_T_prev.dot(Sf);
        }
        const Scalar viscous_dissipation = double_dot(c.tau_prev, c.grad_U_prev) * c.volume;
        const Scalar DpDt = ((c.p_prev - c.p) / TimeStep + c.U_prev.dot(c.grad_p_prev))* c.volume;
        c.rhoh_next = c.rhoh + TimeStep / c.volume * (-convection_flux + diffusion_flux + viscous_dissipation + DpDt);
        c.h_star = c.rhoh_next / c.rho_prev;
        c.T_star = c.h_star / c.specific_heat_p;
    }

    /// Interpolation from cell to face & Apply B.C. for T_star
    for (auto& f : face)
    {
        if (f.at_boundary)
        {
            if (f.parent->T_BC == Dirichlet)
                f.T_star = f.T;
            else if (f.parent->T_BC == Neumann)
            {
                auto c = f.c0 ? f.c0 : f.c1;
                const Vector &r = f.c0 ? f.r0 : f.r1;
                f.T_star = c->T_star + f.grad_T.dot(r);
            }
            else
                throw unsupported_boundary_condition(f.parent->T_BC);
        }
        else
            f.T_star = f.ksi0 * f.c0->T_star + f.ksi1 * f.c1->T_star; /// Less accurate, but bounded
    }

    /// Consistency for h_star
    for(auto &f : face)
    {
        f.h_star = f.specific_heat_p * f.T_star;
    }
}

static void step3()
{
    /// Prediction of density
    for (auto &c : cell)
    {
        c.rho_next = EOS(c.p_prev, c.T_star);
    }
    for(auto &f : face)
    {
        f.rho_next = EOS(f.p_prev, f.T_star);
    }
}

static void step4()
{
    /// Update of temperature
    for (auto &c : cell)
    {
        c.h_next = c.rhoh_next / c.rho_next;
        c.T_next = c.h_next / c.specific_heat_p;
    }

    /// Interpolation from cell to face & Apply B.C. for T_next
    for (auto& f : face)
    {
        if (f.at_boundary)
        {
            if (f.parent->T_BC == Dirichlet)
                f.T_next = f.T;
            else if (f.parent->T_BC == Neumann)
            {
                auto c = f.c0 ? f.c0 : f.c1;
                const Vector &r = f.c0 ? f.r0 : f.r1;
                f.T_next = c->T_next + f.grad_T.dot(r);
            }
            else
                throw unsupported_boundary_condition(f.parent->T_BC);
        }
        else
            f.T_next = f.ksi0 * f.c0->T_next + f.ksi1 * f.c1->T_next; /// Less accurate, but bounded
    }

    /// Consistency for h_next
    for(auto &f : face)
    {
        f.h_next = f.specific_heat_p * f.T_next;
    }
}

static void step5(Scalar TimeStep)
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
            convection_flux += f->rhoU_prev.dot(Sf) * f->U_prev;
            pressure_flux += f->p_prev * Sf;
            viscous_flux += f->tau_prev * Sf;
        }
        c.rhoU_star = c.rhoU + TimeStep / c.volume * (-convection_flux - pressure_flux + viscous_flux);
    }

    /// Interpolation from cell to face & Apply B.C. for rhoU*
    for (auto& f : face)
    {
        if (f.at_boundary)
        {
            Vector U_star(0.0, 0.0, 0.0);
            const auto U_BC = f.parent->U_BC;
            if (U_BC == Dirichlet)
                U_star = f.U;
            else if(U_BC == Neumann)
            {
                auto c = f.c0 ? f.c0 : f.c1;
                const Vector &r = f.c0 ? f.r0 : f.r1;
                U_star = c->U_star + r.transpose() * f.grad_U;
            }
            else
                throw unsupported_boundary_condition(U_BC);

            f.rhoU_star = f.rho_next * U_star;
        }
        else
        {
            f.rhoU_star = f.ksi0 * f.c0->rhoU_star + f.ksi1 * f.c1->rhoU_star;
            /// Momentum interpolation
            const Vector d01 = f.c1->centroid - f.c0->centroid;
            const Vector compact_grad_p = (f.c1->p_prev - f.c0->p_prev) / (d01.dot(d01)) * d01;
            const Vector mean_grad_p = 0.5 * (f.c1->grad_p_prev + f.c0->grad_p_prev);
            f.rhoU_star -= TimeStep * (compact_grad_p - mean_grad_p);
        }
    }
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
        l2 = GRAD_Cell_PressureCorrection();

        /// Interpolate gradient of $p'$ from cell to face
        /// Boundary + Internal
        GRAD_Face_PressureCorrection();

        /// Report
        std::cout << "\n" << SEP;
        std::cout << std::left << std::setw(14) << l1 << "    " << std::setw(26) << l2;

        /// Next loop if needed
        ++cnt;
    }
    return cnt;
}

static void step6(Scalar TimeStep)
{
    std::cout << SEP << "\nSolving pressure-correction ...";
    std::cout << SEP << "\n--------------------------------------------";
    std::cout << SEP << "\n||p'-p'_prev||    ||grad(p')-grad(p')_prev||";
    std::cout << SEP << "\n--------------------------------------------";
    const int poisson_noc_iter = ppe(TimeStep);
    std::cout << SEP << "\n--------------------------------------------";
    std::cout << SEP << "\nConverged after " << poisson_noc_iter << " iterations" << std::endl;

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

    /// Reconstruct gradient of $p'$ at cell centroid
    for (auto& c : cell)
    {
        Vector b(0.0, 0.0, 0.0);
        const size_t Nf = c.surface.size();
        for (size_t j = 0; j < Nf; ++j)
        {
            auto f = c.surface.at(j);
            const Vector &Sf = c.S.at(j);
            b += f->grad_p_prime_sn.dot(Sf) * Sf / f->area;
        }
        c.grad_p_prime = c.TeC_INV * b;
    }
}

static void step7(Scalar TimeStep)
{
    /// rhoU_next on interior face
    for (auto& f : face)
    {
        if (!f.at_boundary)
            f.rhoU_next = f.rhoU_star - TimeStep * f.grad_p_prime_sn;
    }

    /// Update pressure-velocity coupling on cell
    for (auto& C : cell)
    {
        C.rhoU_next = C.rhoU_star - TimeStep * C.grad_p_prime;
        C.U_next = C.rhoU_next / C.rho_next;
        C.p_next = C.p_prev + C.p_prime;
    }

    /// Gradient of U_next on cell
    GRAD_Cell_Velocity_next();

    /// Interpolation from cell to face for U_next
    for (auto &f : face)
    {
        if(f.at_boundary)
        {
            const auto U_BC = f.parent->U_BC;
            if(U_BC == Dirichlet)
                f.U_next = f.U;
            else if (U_BC == Neumann)
            {
                auto C = f.c0 ? f.c0 : f.c1;
                const Vector &r = f.c0 ? f.r0 : f.r1;
                f.U_next = C->U_star + r.transpose() * f.grad_U_next;
            }
            else
                throw unsupported_boundary_condition(U_BC);
        }
        else
        {
            if (f.rhoU_next.dot(f.n01) > 0)
                f.U_next = f.c0->U_next + f.r0.transpose() * f.c0->grad_U_next;
            else
                f.U_next = f.c1->U_next + f.r1.transpose() * f.c1->grad_U_next;
        }
    }

    /// rhoU_next on boundary face
    for (auto& f : face)
    {
        if (f.at_boundary)
            f.rhoU_next = f.rho_next * f.U_next;
    }

    /// gradient of p_next
    GRAD_Cell_Pressure_next();

    /// Interpolation from cell to face for p_next
    for (auto &f : face)
    {
        if(f.at_boundary)
        {
            const auto p_BC = f.parent->p_BC;
            if (p_BC == Dirichlet)
                f.p_next = f.p;
            else if (p_BC == Neumann)
            {
                auto C = f.c0 ? f.c0 : f.c1;
                const Vector &d = f.c0 ? f.r0 : f.r1;
                f.p_next = C->p_next + f.grad_p_next.dot(d);
            }
            else
                throw unsupported_boundary_condition(p_BC);
        }
        else
        {
            const Scalar p_0 = f.c0->p_next + f.c0->grad_p_next.dot(f.r0);
            const Scalar p_1 = f.c1->p_next + f.c1->grad_p_next.dot(f.r1);
            f.p_next = 0.5 * (p_0 + p_1); /// CDS
        }
    }

    /// For next iteration
    for(auto &C : cell)
    {
        C.U_prev = C.U_next;
        C.p_prev = C.p_next;
    }
    for(auto &f : face)
    {
        f.rhoU_prev = f.rhoU_next;
        f.p_prev = f.p_next;
    }
}

static void step8()
{
    /// Update physical properties at centroid of each cell.
    for (auto& c : cell)
    {
        /// Dynamic viscosity
        // c.mu = Sutherland(c.T);
        c.viscosity = c.rho / Re;
    }

    /// Enforce boundary conditions for primitive variables.
    BC_Primitive();

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
        f.viscosity = f.rho / Re;
    }

    /// Viscous shear stress on each face.
    calc_face_viscous_shear_stress();

    /// rhoU on boundary
    for (const auto &e : patch)
        for (auto f : e.surface)
            f->rhoU = f->rho * f->U;
}

/**
 * 1st-order explicit time-marching.
 * Pressure-Velocity coupling is solved using Fractional-Step Method.
 * @param TimeStep
 */
void ForwardEuler(Scalar TimeStep)
{
    /// Prepare m=0
    step1(TimeStep);

    /// Semi-Implicit iteration
    for(int m = 0; m < 3; ++m)
    {
        step2(TimeStep);
        step3();
        step4();
        step5(TimeStep);
        step6(TimeStep); /// Correction Step
        step7(TimeStep);
    }

    /// Store all variables for new time-step
    step8();
}
