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

static int ppe(Scalar TimeStep)
{
    for(auto &c : cell)
    {
        c.p_prime = 0.0;
        c.grad_p_prime.setZero();
    }
    for(auto &f : face)
    {
        f.p_prime = 0.0;
        f.grad_p_prime.setZero();
    }

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

/**
 * 1st-order explicit time-marching.
 * Pressure-Velocity coupling is solved using Fractional-Step Method.
 * @param TimeStep
 */
void ForwardEuler(Scalar TimeStep)
{
    for(auto &f : face)
    {
        f.rhoU_next = f.rhoU;
        f.U_next = f.U;
        f.p_next = f.p;
        f.tau_next = f.tau;
        f.rho_next = f.rho;
        f.h_next = f.h;
        f.grad_T_next = f.grad_T;
        f.T_next = f.T;
        f.rhoh_next = f.rhoh;
    }
    for(auto &C : cell)
    {
        C.p_next = C.p;
        C.grad_p_next = C.grad_p;
        C.T_next = C.T;
        C.rho_next = C.rho;
        C.rhoU_next = C.rhoU;
        C.U_next = C.U;
        C.rhoh_next = C.rhoh;
        C.h_next = C.h;
        C.tau_next = C.tau;
    }

    for (int m=1; m <= 6; ++m)
    {
        std::cout << "\nm=" << m << std::endl;

        /// Update density
        for (auto &c : cell)
            c.rho_next = EOS(c.p_next, c.T_next);

        for (auto &f : face)
            f.rho_next = EOS(f.p_next, f.T_next);

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
                convection_flux += f->rhoU_next.dot(Sf) * f->U_next;
                pressure_flux += f->p_next * Sf;
                viscous_flux += f->tau_next * Sf;
            }
            c.rhoU_star = c.rhoU + TimeStep / c.volume * (-convection_flux - pressure_flux + viscous_flux);
        }

        /// Interpolation from cell to face & Apply B.C. for rhoU*
        for (auto& f : face)
        {
            if (f.at_boundary)
            {
                const auto U_BC = f.parent->U_BC;
                if (U_BC == Dirichlet)
                    f.rhoU_star = f.rho_next * f.U;
                else if(U_BC == Neumann)
                {
                    auto c = f.c0 ? f.c0 : f.c1;
                    const Vector &r = f.c0 ? f.r0 : f.r1;
                    f.rhoU_star = c->rhoU_star + f.rho_next * (r.transpose() * f.grad_U).transpose();
                }
                else
                    throw unsupported_boundary_condition(U_BC);
            }
            else
            {
                f.rhoU_star = f.ksi0 * f.c0->rhoU_star + f.ksi1 * f.c1->rhoU_star;
                /// Momentum interpolation
                const Vector d01 = f.c1->centroid - f.c0->centroid;
                const Vector compact_grad_p = (f.c1->p_next - f.c0->p_next) / (d01.dot(d01)) * d01;
                const Vector mean_grad_p = 0.5 * (f.c1->grad_p_next + f.c0->grad_p_next);
                f.rhoU_star -= TimeStep * (compact_grad_p - mean_grad_p);
            }
        }

        /// Perturbation of EOS
        std::vector<Scalar> unsteady_diagonal(NumOfCell, 0.0);
        for(size_t i = 0; i < NumOfCell; ++i)
        {
            const auto &C = cell.at(i);
            unsteady_diagonal.at(i) = C.volume / (TimeStep * TimeStep * 287.7 * C.T_next);
        }

        calcPressureCorrectionEquationCoef(A_dp_2, unsteady_diagonal);
        prepare_dp_solver(A_dp_2, dp_solver_2);

        std::cout << "\nSolving pressure-correction ...";
        std::cout << "\n--------------------------------------------";
        std::cout << "\n||p'-p'_prev||    ||grad(p')-grad(p')_prev||";
        std::cout << "\n--------------------------------------------";
        const auto noc_iter = ppe(TimeStep);
        std::cout << "\n--------------------------------------------";
        std::cout << "\nConverged after " << noc_iter << " iterations" << std::endl;

        check_bound<Cell>("p'_C", cell, [](const Cell& C){return C.p_prime;});

        /// Calculate $\frac{\partial p'}{\partial n}$ on face.
        /// Should be CONSISTENT with NOC method used in 'ppe' !!!
        INTERP_Face_snGrad_PressureCorrection();

        /// Smooth gradient of $p'$ on cell.
        /// For stability reason mostly.
        //RECONST_Cell_Grad_PressureCorrection();

        /// Update pressure
        for (auto& C : cell)
            C.p_next += + C.p_prime;

        /// Interpolation from cell to face & Apply B.C. for p
        GRAD_Cell_Pressure_next();
        GRAD_Face_Pressure_next();
        INTERP_Face_Pressure_next();

        /// Update density
//        for (auto& C : cell)
//            C.rho_next = EOS(C.p_next, C.T_next);
//
//        for (auto& f : face)
//            f.rho_next = EOS(f.p_next, f.T_next);

        /// Update mass flux on cell
        for (auto& C : cell)
        {
            C.rhoU_next = C.rhoU_star - TimeStep * C.grad_p_prime;
            C.U_next = C.rhoU_next / C.rho_next;
        }

        /// Interpolation from cell to face & Apply B.C. for U
        GRAD_Cell_Velocity_next();
        GRAD_Face_Velocity_next();
        INTERP_Face_Velocity_next();

        /// Update mass flux on face
        for (auto& f : face)
        {
            if (f.at_boundary)
                f.rhoU_next = f.rho_next * f.U_next;
            else
                f.rhoU_next = f.rhoU_star - TimeStep * f.grad_p_prime_sn;
        }

        /// Update viscous shear stress
        CALC_Cell_ViscousShearStress_next();
        CALC_Face_ViscousShearStress_next();

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
                convective_flux += f->rhoU_next.dot(Sf) * f->h_next;
                diffusive_flux += f->conductivity * f->grad_T_next.dot(Sf);
            }
            //const Scalar viscous_dissipation = double_dot(c.tau_prev, c.grad_U_prev) * c.volume;
            //const Scalar DpDt = ((c.p_prev - c.p) / TimeStep + c.U_prev.dot(c.grad_p_prev))* c.volume;
            //c.rhoh_next = c.rhoh + TimeStep / c.volume * (-convective_flux + diffusive_flux + viscous_dissipation + DpDt);
            c.rhoh_next = c.rhoh + TimeStep / c.volume * (-convective_flux + diffusive_flux);
            c.h_next = c.rhoh_next / c.rho_next;
            c.T_next = c.h_next / c.specific_heat_p;
        }

        /// Interpolation from cell to face & Apply B.C. for T
        GRAD_Cell_Temperature_next();
        GRAD_Face_Temperature_next();
        INTERP_Face_Temperature_next();

        for (auto &f : face)
        {
            f.h_next = f.specific_heat_p * f.T;
            f.rhoh_next = f.rho_next * f.h_next;
        }

        check_bound<Face>("T_f@(next)", face, [](const Face& f){return f.T_next;});
        check_bound<Cell>("T_C@(next)", cell, [](const Cell& C){return C.T_next;});

        check_bound<Face>("|grad(T)|_f@(next)", face, [](const Face& f){return f.grad_T_next.norm();});
        check_bound<Cell>("|grad(T)|_C@(next)", cell, [](const Cell& C){return C.grad_T_next.norm();});
    }

    for(auto &f : face)
    {
        f.rhoU = f.rhoU_next;
        f.U = f.U_next;
        f.p = f.p_next;
        f.tau = f.tau_next;
        f.rho = f.rho_next;
        f.h = f.h_next;
        f.grad_T = f.grad_T_next;
        f.T = f.T_next;
        f.rhoh = f.rhoh_next;
    }
    for(auto &C : cell)
    {
        C.p = C.p_next;
        C.grad_p = C.grad_p_next;
        C.T = C.T_next;
        C.rho = C.rho_next;
        C.rhoU = C.rhoU_next;
        C.U = C.U_next;
        C.rhoh = C.rhoh_next;
        C.h = C.h_next;
        C.grad_T = C.grad_T_next;
        C.tau = C.tau_next;
    }

    BC_Primitive();

    /// Property @(n+1), Cell & Face(Boundary+Internal)
    CALC_Cell_Viscosity();
    CALC_Cell_SpecificHeat();
    CALC_Cell_Conductivity();
    CALC_Face_Viscosity();
    CALC_Face_SpecificHeat();
    CALC_Face_Conductivity();

    INTERP_Node_Primitive();
}
