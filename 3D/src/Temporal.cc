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
extern SX_MAT A_dp_2;
extern SX_VEC Q_dp_2;
extern SX_VEC x_dp_2;
extern SX_AMG dp_solver_2;

template<typename T>
static void check_bound
(
    const std::string& nm,
    const NaturalArray<T>& grp,
    const std::function<Scalar(const T&)> &extractor
)
{
    Scalar var_min = extractor(grp.at(0));
    Scalar var_max = var_min;
    for (size_t i = 1; i < grp.size(); ++i)
    {
        const auto cur_var = extractor(grp.at(i));
        if (cur_var < var_min)
            var_min = cur_var;
        if (cur_var > var_max)
            var_max = cur_var;
    }
    std::cout << nm << " : " << var_min << " ~ " << var_max << std::endl;
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
        c.tau_prev = c.tau;
        c.grad_U_prev = c.grad_U;
        c.grad_p_prev = c.grad_p;
    }

    for (auto &f : face)
    {
        f.U_prev = f.U;
        f.p_prev = f.p;
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
        const Scalar viscous_dissipation = 0;//double_dot(c.tau_prev, c.grad_U_prev) * c.volume;
        const Scalar DpDt = 0;// ((c.p_prev - c.p) / TimeStep + c.U_prev.dot(c.grad_p_prev))* c.volume;
        c.rhoh_next = c.rhoh + TimeStep / c.volume * (-convection_flux + diffusion_flux + viscous_dissipation + DpDt);
        c.h_star = c.rhoh_next / c.rho_prev;
        c.T_star = c.h_star / c.specific_heat_p;
    }

    /// Interpolation from cell to face & Apply B.C. for T_star
    GRAD_Cell_Temperature_star();
    INTERP_Face_Temperature_star();
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
    /// Update temperature
    for (auto &c : cell)
    {
        c.h_next = c.rhoh_next / c.rho_next;
        c.T_next = c.h_next / c.specific_heat_p;
    }

    /// Interpolation from cell to face & Apply B.C. for T_next
    GRAD_Cell_Temperature_next();
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
        c.p_prime = 0.0;
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
        std::cout << "\n" << std::left << std::setw(14) << l1 << "    " << std::setw(26) << l2;

        /// Next loop if needed
        ++cnt;
        if (cnt > 20)
            break;
    }
    return cnt;
}

static void step6(Scalar TimeStep)
{
    /// Corrector
    std::cout << "\nSolving pressure-correction ...";
    std::cout << "\n--------------------------------------------";
    std::cout << "\n||p'-p'_prev||    ||grad(p')-grad(p')_prev||";
    std::cout << "\n--------------------------------------------";
    const auto noc_iter = ppe(TimeStep);
    std::cout << "\n--------------------------------------------";
    std::cout << "\nConverged after " << noc_iter << " iterations";
    std::cout << "\n" << std::endl;

    /// Calculate $\frac{\partial p'}{\partial n}$ on face.
    /// Should be CONSISTENT with NOC method used in 'ppe' !!!
    INTERP_Face_snGrad_PressureCorrection();

    /// Smooth gradient of $p'$ on cell.
    /// For stability reason mostly.
    RECONST_Cell_Grad_PressureCorrection();
}

static void step7(Scalar TimeStep)
{
    /// Update mass flux on internal face
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

    /// Interpolation from cell to face & Apply B.C. for U_next
    GRAD_Cell_Velocity_next();
    GRAD_Face_Velocity_next();
    INTERP_Face_Velocity_next();

    /// Update viscous shear stress
    CALC_Cell_ViscousShearStress_next();
    CALC_Face_ViscousShearStress_next();

    /// Update mass flux on boundary face
    for (auto& f : face)
    {
        if (f.at_boundary)
            f.rhoU_next = f.rho_next * f.U_next;
    }

    /// Interpolation from cell to face & Apply B.C. for p_next
    GRAD_Cell_Pressure_next();
    INTERP_Face_Pressure_next();
}

static void aux()
{
    for(auto &C : cell)
    {
        C.tau_prev = C.tau_next;
        C.grad_U_prev = C.grad_U_next;
        C.p_prev = C.p_next;
        C.grad_p_prev = C.grad_p_next;
        C.U_prev = C.U_next;
        C.rho_prev = C.rho_next;
    }
    for(auto &f : face)
    {
        f.rhoU_prev = f.rhoU_next;
        f.h_prev = f.h_next;
        f.grad_T_prev = f.grad_T_next;
        f.p_prev = f.p_next;
        f.U_prev = f.U_next;
        f.tau_prev = f.tau_next;
    }
}

static void step8()
{
    for (auto &f : face)
    {
        if (!f.at_boundary)
        {
            f.rho = f.rho_next;
            f.U = f.U_next;
            f.p = f.p_next;
            f.T = f.T_next;
            f.h = f.h_next;
            f.rhoU = f.rhoU_next;
            f.rhoh = f.rhoh_next;
        }
    }

    for (auto &C : cell)
    {
        C.rho = C.rho_next;
        C.U = C.U_next;
        C.p = C.p_next;
        C.T = C.T_next;
        C.h = C.h_next;
        C.rhoU = C.rhoU_next;
        C.rhoh = C.rhoh_next;
    }

    /// Enforce B.C. for {$\vec{U}$, $p$, $T$} @(n+1), Face(Boundary)
    BC_Primitive();

    /// {$\nabla \vec{U}$, $\nabla p$, $\nabla T$} @(n+1), Cell & Face(Boundary+Internal)
    GRAD_Cell_Velocity();
    GRAD_Cell_Pressure();
    GRAD_Cell_Temperature();
    GRAD_Face_Velocity();
    GRAD_Face_Pressure();
    GRAD_Face_Temperature();

    /// {$\vec{U}$, $p$, $T$} @(n+1), Face(Boundary)
    INTERP_BoundaryFace_Velocity();
    INTERP_BoundaryFace_Pressure();
    INTERP_BoundaryFace_Temperature();

    /// {$\rho$} @(n+1), Face(Boundary)
    for (auto &f : face)
    {
        if(f.at_boundary)
        {
            f.rho = EOS(f.p, f.T);
        }
    }

    /// {$\nabla \rho$} @(n+1), Cell & Face(Boundary+Internal)
    GRAD_Cell_Density();
    GRAD_Face_Density();

    /// {$\rho$, $\vec{U}$, $p$, $T$} @(n+1), Node
    INTERP_Node_Primitive();

    /// Property @(n+1), Cell & Face(Boundary+Internal)
    CALC_Cell_Viscosity();
    CALC_Cell_SpecificHeat();
    CALC_Cell_Conductivity();
    CALC_Face_Viscosity();
    CALC_Face_SpecificHeat();
    CALC_Face_Conductivity();

    /// {$\tau$} @(n+1), Cell & Face(Boundary+Internal)
    CALC_Cell_ViscousShearStress();
    CALC_Face_ViscousShearStress();

    /// {$h$, $\rho h$, $\rho \vec{U}$} @(n+1), Face(Boundary)
    for (auto &f : face)
    {
        if (f.at_boundary)
        {
            f.h = Enthalpy(f.specific_heat_p, f.T);
            f.rhoh = f.rho * f.h;
            f.rhoU = f.rho * f.U;
        }
    }
}

/**
 * 1st-order explicit time-marching.
 * Pressure-Velocity coupling is solved using Fractional-Step Method.
 * @param TimeStep
 */
void ForwardEuler(Scalar TimeStep)
{
    /// Prepare m=0
    std::cout << "\nm=0" << std::endl;
    check_bound<Cell>("mu_C@(n)", cell, [](const Cell &c) { return c.viscosity; });
    check_bound<Cell>("lambda_C@(n)", cell, [](const Cell &c) { return c.conductivity; });
    check_bound<Cell>("Cp_C@(n)", cell, [](const Cell &c) { return c.specific_heat_p; });
    check_bound<Cell>("Cv_C@(n)", cell, [](const Cell &c) { return c.specific_heat_v; });
    check_bound<Cell>("rho_C@(n)", cell, [](const Cell &c) { return c.rho; });
    check_bound<Cell>("u_C@(n)", cell, [](const Cell &c) { return c.U.x(); });
    check_bound<Cell>("v_C@(n)", cell, [](const Cell &c) { return c.U.y(); });
    check_bound<Cell>("w_C@(n)", cell, [](const Cell &c) { return c.U.z(); });
    check_bound<Cell>("p_C@(n)", cell, [](const Cell &c) { return c.p; });
    check_bound<Cell>("T_C@(n)", cell, [](const Cell &c) { return c.T; });
    check_bound<Cell>("h_C@(n)", cell, [](const Cell &c) { return c.h; });
    check_bound<Cell>("rhou_C@(n)", cell, [](const Cell &c) { return c.rhoU.x(); });
    check_bound<Cell>("rhov_C@(n)", cell, [](const Cell &c) { return c.rhoU.y(); });
    check_bound<Cell>("rhow_C@(n)", cell, [](const Cell &c) { return c.rhoU.z(); });
    check_bound<Cell>("rhoh_C@(n)", cell, [](const Cell &c) { return c.rhoh; });
    std::cout << std::endl;
    check_bound<Face>("mu_f@(n)", face, [](const Face &f) { return f.viscosity; });
    check_bound<Face>("lambda_f@(n)", face, [](const Face &f) { return f.conductivity; });
    check_bound<Face>("Cp_f@(n)", face, [](const Face &f) { return f.specific_heat_p; });
    check_bound<Face>("Cv_f@(n)", face, [](const Face &f) { return f.specific_heat_v; });
    check_bound<Face>("rho_f@(n)", face, [](const Face &f) { return f.rho; });
    check_bound<Face>("u_f@(n)", face, [](const Face &f) { return f.U.x(); });
    check_bound<Face>("v_f@(n)", face, [](const Face &f) { return f.U.y(); });
    check_bound<Face>("w_f@(n)", face, [](const Face &f) { return f.U.z(); });
    check_bound<Face>("p_f@(n)", face, [](const Face &f) { return f.p; });
    check_bound<Face>("T_f@(n)", face, [](const Face &f) { return f.T; });
    check_bound<Face>("h_f@(n)", face, [](const Face &f) { return f.h; });
    check_bound<Face>("rhou_f@(n)", face, [](const Face &f) { return f.rhoU.x(); });
    check_bound<Face>("rhov_f@(n)", face, [](const Face &f) { return f.rhoU.y(); });
    check_bound<Face>("rhow_f@(n)", face, [](const Face &f) { return f.rhoU.z(); });
    check_bound<Face>("rhoh_f@(n)", face, [](const Face &f) { return f.rhoh; });
    step1(TimeStep);

    /// Semi-Implicit iteration
    for(int m = 1; m <= 1; ++m)
    {
        std::cout << "\nm=" << m << std::endl;
        check_bound<Cell>("rho_C@(m-1)", cell, [](const Cell &c) { return c.rho_prev; });

        /// {$\rho h$} @(m+1), Cell
        /// {$h$, $T$} @(*), Cell & Face(Boundary+Internal)
        /// {$\nabla T$} @(*), Cell
        step2(TimeStep);
        check_bound<Cell>("rhoh_C@(m+1)", cell, [](const Cell &c) { return c.rhoh_next; });
        check_bound<Cell>("h_C@(*)", cell, [](const Cell &c) { return c.h_star; });
        check_bound<Cell>("T_C@(*)", cell, [](const Cell &c) { return c.T_star; });
        check_bound<Face>("T_f@(*)", face, [](const Face &f) { return f.T_star; });

        /// {$\rho$} @(m+1), Cell & Face(Boundary+Internal)
        step3();
        check_bound<Cell>("rho_C@(m+1)", cell, [](const Cell &c) { return c.rho_next; });
        check_bound<Face>("rho_f@(m+1)", face, [](const Face &f) { return f.rho_next; });

        /// {$h$, $T$} @(m+1), Cell & Face(Boundary+Internal)
        /// {$\nabla T$} @(m+1), Cell
        step4();
        check_bound<Cell>("h_C@(m+1)", cell, [](const Cell &c) { return c.h_next; });
        check_bound<Cell>("T_C@(m+1)", cell, [](const Cell &c) { return c.T_next; });
        check_bound<Face>("h_f@(m+1)", face, [](const Face &f) { return f.h_next; });
        check_bound<Face>("T_f@(m+1)", face, [](const Face &f) { return f.T_next; });

        /// {$\rho \vec{U}$} @(*), Cell & Face(Boundary+Internal)
        step5(TimeStep);
        check_bound<Cell>("rhou_C@(*)", cell, [](const Cell &c) { return c.rhoU_star.x(); });
        check_bound<Cell>("rhov_C@(*)", cell, [](const Cell &c) { return c.rhoU_star.y(); });
        check_bound<Cell>("rhow_C@(*)", cell, [](const Cell &c) { return c.rhoU_star.z(); });
        check_bound<Face>("rhou_f@(*)", face, [](const Face &f) { return f.rhoU_star.x(); });
        check_bound<Face>("rhov_f@(*)", face, [](const Face &f) { return f.rhoU_star.y(); });
        check_bound<Face>("rhow_f@(*)", face, [](const Face &f) { return f.rhoU_star.z(); });

        /// {$p'$}, Cell
        /// {$\nabla p'$}, Cell & Face(Boundary+Internal)
        /// {$\frac{\partial p'}{\partial n}$}, Face(Boundary+Internal)
        step6(TimeStep);
        check_bound<Cell>("p'_C", cell, [](const Cell &c) { return c.p_prime; });
        check_bound<Cell>("grad_x(p')_C", cell, [](const Cell &c) { return c.grad_p_prime.x(); });
        check_bound<Cell>("grad_y(p')_C", cell, [](const Cell &c) { return c.grad_p_prime.y(); });
        check_bound<Cell>("grad_z(p')_C", cell, [](const Cell &c) { return c.grad_p_prime.z(); });
        check_bound<Face>("grad_x(p')_f", face, [](const Face &f) { return f.grad_p_prime.x(); });
        check_bound<Face>("grad_y(p')_f", face, [](const Face &f) { return f.grad_p_prime.y(); });
        check_bound<Face>("grad_z(p')_f", face, [](const Face &f) { return f.grad_p_prime.z(); });

        /// {$p$, $\vec{U}$, $\rho \vec{U}$} @(m+1), Cell & Face(Boundary+Internal)
        /// {$\nabla \vec{U}$}, Cell & Face(Boundary+Internal)
        /// {$\tau$} @(m+1), Cell & Face(Boundary+Internal)
        /// {$\nabla p$} @(m+1), Cell
        step7(TimeStep);
        check_bound<Cell>("rhou_C@(m+1)", cell, [](const Cell &c) { return c.rhoU_next.x(); });
        check_bound<Cell>("rhov_C@(m+1)", cell, [](const Cell &c) { return c.rhoU_next.y(); });
        check_bound<Cell>("rhow_C@(m+1)", cell, [](const Cell &c) { return c.rhoU_next.z(); });
        check_bound<Cell>("u_C@(m+1)", cell, [](const Cell &c) { return c.U_next.x(); });
        check_bound<Cell>("v_C@(m+1)", cell, [](const Cell &c) { return c.U_next.y(); });
        check_bound<Cell>("w_C@(m+1)", cell, [](const Cell &c) { return c.U_next.z(); });
        check_bound<Cell>("p_C@(m+1)", cell, [](const Cell &c) { return c.p_next; });
        check_bound<Face>("rhou_f@(m+1)", face, [](const Face &f) { return f.rhoU_next.x(); });
        check_bound<Face>("rhov_f@(m+1)", face, [](const Face &f) { return f.rhoU_next.y(); });
        check_bound<Face>("rhow_f@(m+1)", face, [](const Face &f) { return f.rhoU_next.z(); });
        check_bound<Face>("u_f@(m+1)", face, [](const Face &f) { return f.U_next.x(); });
        check_bound<Face>("v_f@(m+1)", face, [](const Face &f) { return f.U_next.y(); });
        check_bound<Face>("w_f@(m+1)", face, [](const Face &f) { return f.U_next.z(); });
        check_bound<Face>("p_f@(m+1)", face, [](const Face &f) { return f.p_next; });

        /// For next iteration
        aux();
    }

    /// Store all variables for new time-step
    step8();
}
