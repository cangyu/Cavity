#include <iostream>
#include <fstream>
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
        c.rho_prev = c.rho;// + TimeStep / c.volume * (-convection_flux);
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

static int ppe(Scalar TimeStep)
{
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
    }

    sx_mat_destroy(&A_dp_2);

    return cnt;
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
    BC_Primitive();

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
            convection_flux += f->rhoU.dot(Sf) * f->U;
            pressure_flux += f->p * Sf;
            viscous_flux += f->tau * Sf;
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
                U_star = c->rhoU_star / c->rho + (r.transpose() * f.grad_U).transpose();
            }
            else
                throw unsupported_boundary_condition(U_BC);

            f.rhoU_star = f.rho * U_star;
        }
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

    /// Perturbation of EOS
    std::vector<Scalar> unsteady_diagonal(NumOfCell, 0.0);
    for(size_t i = 0; i < NumOfCell; ++i)
    {
        const auto &C = cell.at(i);
        unsteady_diagonal.at(i) = C.volume / (TimeStep * TimeStep * 287.7 * C.T);
    }

    calcPressureCorrectionEquationCoef(A_dp_2, unsteady_diagonal);
    prepare_dp_solver(A_dp_2, dp_solver_2);

    /// Solve p'
    for(auto &c : cell)
    {
        c.p_prime = 0.0;
        c.grad_p_prime.setZero();
    }
    for(auto &f : face)
        f.grad_p_prime.setZero();

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

    /// Update mass flux on internal face
    for (auto& f : face)
    {
        if (!f.at_boundary)
            f.rhoU = f.rhoU_star - TimeStep * f.grad_p_prime_sn;
    }

    /// Update pressure-velocity coupling on cell
    for (auto& C : cell)
    {
        C.rhoU = C.rhoU_star - TimeStep * C.grad_p_prime;
        C.p += C.p_prime;
        C.rho = EOS(C.p, C.T);
        C.U = C.rhoU / C.rho;
    }

    /// Interpolation from cell to face & Apply B.C. for p
    GRAD_Cell_Pressure();
    GRAD_Face_Pressure();
    INTERP_Face_Pressure();

    for (auto& f : face)
    {
        f.rho = EOS(f.p, f.T);
        if (!f.at_boundary)
            f.U = f.rhoU / f.rho;
    }

    /// Interpolation from cell to face & Apply B.C. for U
    GRAD_Cell_Velocity();
    GRAD_Face_Velocity();
    INTERP_BoundaryFace_Velocity();

    /// Update viscous shear stress
    CALC_Cell_ViscousShearStress();
    CALC_Face_ViscousShearStress();

    /// Update mass flux on boundary face
    for (auto& f : face)
    {
        if (f.at_boundary)
            f.rhoU = f.rho * f.U;
    }

    /// Prediction of energy
    for(auto &c : cell)
    {
        Scalar convective_flux = 0.0;
        Scalar diffusive_flux = 0.0;
        const auto Nf = c.S.size();
        for (int j = 0; j < Nf; ++j)
        {
            auto f = c.surface.at(j);
            const auto &Sf = c.S.at(j);
            convective_flux += f->rhoU.dot(Sf) * f->h;
            diffusive_flux += f->conductivity * f->grad_T.dot(Sf);
        }
        //const Scalar viscous_dissipation = double_dot(c.tau_prev, c.grad_U_prev) * c.volume;
        //const Scalar DpDt = ((c.p_prev - c.p) / TimeStep + c.U_prev.dot(c.grad_p_prev))* c.volume;
        //const Scalar total_flux = -c.energy_convective_flux + c.energy_diffusive_flux + viscous_dissipation + DpDt;
        c.rhoh = c.rhoh + TimeStep / c.volume * (-convective_flux + diffusive_flux);
        c.h = c.rhoh / c.rho;
        c.T = c.h / c.specific_heat_p;
    }

    /// Interpolation from cell to face & Apply B.C. for T
    GRAD_Cell_Temperature();
    GRAD_Face_Temperature();
    INTERP_Face_Temperature();

    /// Consistency for h
    for(auto &f : face)
    {
        f.h = f.specific_heat_p * f.T;
    }

    /// Update density
    for (auto &c : cell)
    {
        c.rho = EOS(c.p, c.T);
    }
    for(auto &f : face)
    {
        f.rho = EOS(f.p, f.T);
    }

    /// Property @(n+1), Cell & Face(Boundary+Internal)
    CALC_Cell_Viscosity();
    CALC_Cell_SpecificHeat();
    CALC_Cell_Conductivity();
    CALC_Face_Viscosity();
    CALC_Face_SpecificHeat();
    CALC_Face_Conductivity();

    INTERP_Node_Primitive();
}
