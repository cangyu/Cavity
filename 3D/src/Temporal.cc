#include <iostream>
#include <iomanip>
#include "../inc/MISC.h"
#include "../inc/BC.h"
#include "../inc/PoissonEqn.h"
#include "../inc/CHEM.h"
#include "../inc/Gradient.h"
#include "../inc/Spatial.h"
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
    INTERP_Face_Temperature_star();

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
    INTERP_Face_Temperature_next();

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
    INTERP_Face_MassFlux_star(TimeStep);
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

    /// Calculate $\frac{\partial p'}{\partial n}$ on face.
    /// Should be CONSISTENT with NOC method used in 'ppe' !!!
    INTERP_Face_snGrad_PressureCorrection();

    /// Smooth gradient of $p'$ on cell.
    /// For stability reason mostly.
    RECONST_Cell_Grad_PressureCorrection();
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
    INTERP_Face_Velocity_next();

    /// rhoU_next on boundary face
    for (auto& f : face)
    {
        if (f.at_boundary)
            f.rhoU_next = f.rho_next * f.U_next;
    }

    /// gradient of p_next
    GRAD_Cell_Pressure_next();

    /// Interpolation from cell to face for p_next
    INTERP_Face_Pressure_next();

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
