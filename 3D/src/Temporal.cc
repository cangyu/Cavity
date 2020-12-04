#include <iostream>
#include <iomanip>
#include "../inc/Miscellaneous.h"
#include "../inc/BC.h"
#include "../inc/PoissonEqn.h"
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
extern SX_VEC A_dp_2_diag;
extern SX_VEC A_dp_2_diag_unsteady;
extern SX_AMG dp_solver_2;
extern std::string SEP;
extern std::ostream& LOG_OUT;

static const Scalar GAMMA = 1.4;
static const Scalar Pr = 0.72;
static const Scalar Rg = 287.7; // J / (Kg * K)

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
    LOG_OUT << nm << " : " << var_min << " ~ " << var_max << std::endl;
}

static int solve_pressure_correction(Scalar TimeStep)
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
    Scalar err1 = 1.0, err2 = 1.0; /// Convergence monitor
    while (err1 > 1e-10 && err2 > 1e-8)
    {
        /// Solve p' at cell centroid
        PC_updateRHS(&Q_dp_2, TimeStep);
        sx_solver_amg_solve(&dp_solver_2, &x_dp_2, &Q_dp_2);
        err1 = 0.0;
        for (int i = 0; i < NumOfCell; ++i)
        {
            auto& c = cell.at(i);
            const Scalar new_val = sx_vec_get_entry(&x_dp_2, i);
            err1 += std::fabs(new_val - c.p_prime);
            c.p_prime = new_val;
        }
        err1 /= NumOfCell;

        /// Calculate gradient of $p'$ at cell centroid
        err2 = calc_cell_pressure_correction_gradient();

        /// Interpolate gradient of $p'$ from cell centroid to face centroid
        calc_face_pressure_correction_gradient();

        /// Interpolate $p'$ at face centroid
        calc_face_pressure_correction();

        /// Report
        LOG_OUT << SEP << std::left << std::setw(14) << err1 << "    " << std::setw(26) << err2 << std::endl;

        /// Next loop
        ++cnt;
    }
    return cnt;
}

void ForwardEuler(Scalar TimeStep)
{
    check_bound<Cell>("rho_C@(n)", cell, [](const Cell &c) { return c.rho; });
    check_bound<Cell>("p_C@(n)", cell, [](const Cell &c) { return c.p; });
    check_bound<Cell>("T_C@(n)", cell, [](const Cell &c) { return c.T; });

    /// mu, Cp, Cv, lambda @(n)
    for (auto& c : cell)
    {
        c.viscosity = c.rho / Re;
        c.specific_heat_p = 3.5 * Rg;
        c.specific_heat_v = c.specific_heat_p / GAMMA;
        c.conductivity = c.specific_heat_p * c.viscosity / Pr;
    }
    /// B.C. for rho, U, p, T @(n)
    set_bc_of_primitive_var();
    /// grad_rho, grad_U, grad_p, grad_T @(n)
    calc_cell_primitive_gradient();
    /// grad_rho, grad_U, grad_p, grad_T @(n), boundary + internal
    calc_face_primitive_gradient();
    /// rho, U, p, T @(n), boundary + internal
    calc_face_primitive_var();
    check_bound<Face>("rho_f@(n)", face, [](const Face &f) { return f.rho; });
    check_bound<Face>("p_f@(n)", face, [](const Face &f) { return f.p; });
    check_bound<Face>("T_f@(n)", face, [](const Face &f) { return f.T; });
    /// rhoU @(n), boundary
    for (const auto &e : patch)
    {
        for (auto f : e.surface)
            f->rhoU = f->rho * f->U;
    }
    /// h @(n), boundary + internal
    for(auto &f : face)
    {
        f.h = f.specific_heat_p * f.T;
    }
    /// mu, Cp, Cv, lambda @(n), boundary + internal
    for (auto& f : face)
    {
        f.viscosity = f.rho / Re;
        f.specific_heat_p = 3.5 * Rg;
        f.specific_heat_v = f.specific_heat_v / GAMMA;
        f.conductivity = f.specific_heat_p * f.viscosity / Pr;
    }
    /// tau @(n), boundary + internal
    calc_face_viscous_shear_stress();
    /// B.C. for p'
    set_bc_of_pressure_correction();

    /// Init @(m-1)
    LOG_OUT << "m=0" << std::endl;
    for (auto& c : cell)
    {
        c.rho_prev = c.rho;
        c.p_prev = c.p;
        c.T_prev = c.T;
    }
    for (auto& f : face)
    {
        f.rho_prev = f.rho;
        f.p_prev = f.p;
        f.T_prev = f.T;
    }

    /// Loop @(m)
    int m = 0;
    while(++m < 3)
    {
        LOG_OUT << "m=" << m << std::endl;
        if(m >= 2)
        {
            for (auto& c : cell)
            {
                c.rho_prev = c.rho_next;
                c.p_prev = c.p_next;
                c.T_prev = c.T_next;
            }
            for (auto& f : face)
            {
                f.rho_prev = f.rho_next;
                f.p_prev = f.p_next;
                f.T_prev = f.T_next;
            }
        }

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
        for(auto &f : face)
        {
            f.drhodp_prev = 1.0 / (Rg * f.T_prev);
        }
        check_bound<Cell>("drhodp_C@(m-1)", cell, [](const Cell &c) { return c.drhodp_prev; });
        check_bound<Face>("drhodp_f@(m-1)", face, [](const Face &f) { return f.drhodp_prev; });

        /// Contribution to the diagonal of Poisson equation
        for(auto &c : cell)
        {
            const size_t idx = c.index - 1;
            const Scalar val = c.volume * c.drhodp_prev / std::pow(TimeStep, 2);
            sx_vec_set_entry(&A_dp_2_diag_unsteady, idx, val);
        }

        /// Update coefficient of Poisson equation
        PC_updateDiagonalPart(&A_dp_2, &A_dp_2_diag, &A_dp_2_diag_unsteady);

        /// Correction Step
        LOG_OUT << "\n" << SEP << "Solving pressure-correction ..." << std::endl;
        LOG_OUT << SEP << "--------------------------------------------" << std::endl;
        LOG_OUT << SEP << "||p'-p'_prev||    ||grad(p')-grad(p')_prev||" << std::endl;
        LOG_OUT << SEP << "--------------------------------------------" << std::endl;
        const int poisson_noc_iter = solve_pressure_correction(TimeStep);
        LOG_OUT << SEP << "--------------------------------------------" << std::endl;
        LOG_OUT << SEP << "Converged after " << poisson_noc_iter << " iterations" << std::endl;
        check_bound<Cell>("p'_C", cell, [](const Cell &c) { return c.p_prime; });
        check_bound<Face>("p'_f", face, [](const Face &f) { return f.p_prime; });

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
        /*
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
         */

        /// Update
        for (auto& f : face)
        {
            const Scalar rho_prime = f.drhodp_prev * f.p_prime;
            f.rho_star = f.rho_prev + rho_prime;

            if (f.at_boundary)
            {
                f.U_next = f.U;

            }
            else
            {
                const Vector U_prime = - TimeStep * f.sn_grad_p_prime / f.rho_prev;
                f.U_next = f.U_star + U_prime;
                f.rhoU_next = f.rhoU_star + f.rho_prev * U_prime + rho_prime * f.U_star;
            }
            f.p_next = f.p_prev + f.p_prime;
        }
        check_bound<Face>("rho*_f", face, [](const Face &f) { return f.rho_star; });
        check_bound<Face>("p_f@(m)", face, [](const Face &f) { return f.p_next; });

        for (auto& c : cell)
        {
            const Scalar rho_prime = c.drhodp_prev * c.p_prime;
            c.rho_star = c.rho_prev + rho_prime;
            const Vector U_prime = - TimeStep * c.grad_p_prime / c.rho_prev;
            c.U_next = c.U_star + U_prime;
            c.rhoU_next = c.rhoU_star + c.rho_prev * U_prime + rho_prime * c.U_star;
            c.p_next = c.p_prev + c.p_prime;
        }
        check_bound<Cell>("rho*_C", cell, [](const Cell &c) { return c.rho_star; });
        check_bound<Cell>("p_C@(m)", cell, [](const Cell &c) { return c.p_next; });

        /// grad_U @(m)
        calc_cell_velocity_gradient_next();
        /// grad_p @(m)
        calc_cell_pressure_gradient_next();
        /// tau @(m)
        calc_cell_viscous_shear_stress_next();

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
        check_bound<Cell>("rho_C@(m)", cell, [](const Cell &c) { return c.rho_next; });
        check_bound<Cell>("T_C@(m)", cell, [](const Cell &c) { return c.T_next; });

        /// grad_T @(m)
        calc_cell_temperature_gradient_next();
        calc_face_temperature_gradient_next();

        /// T @(m)
        calc_face_temperature_next();
        check_bound<Face>("T_f@(m)", face, [](const Face &f) { return f.T_next; });

        /// rho @(m)
        for(auto &f : face)
        {
            if(f.at_boundary)
                f.rho_next = f.p_next / (Rg * f.T);
            else
                f.rho_next = f.p_next / (Rg * f.T_next);
        }
        check_bound<Face>("rho_f@(m)", face, [](const Face &f) { return f.rho_next; });
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
