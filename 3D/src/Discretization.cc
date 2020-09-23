#include <iostream>
#include "../inc/BC.h"
#include "../inc/PoissonEqn.h"
#include "../inc/CHEM.h"
#include "../inc/Gradient.h"
#include "../inc/Flux.h"
#include "../inc/Discretization.h"

extern int NumOfPnt, NumOfFace, NumOfCell;
extern NaturalArray<Point> pnt;
extern NaturalArray<Face> face;
extern NaturalArray<Cell> cell;
extern NaturalArray<Patch> patch;
extern int NOC_ITER;
extern Scalar Re;
extern SX_MAT A_dp_2;
extern SX_VEC Q_dp_2;
extern SX_VEC x_dp_2;
extern SX_AMG dp_solver_2;

/*********************************************** Spatial Discretization ***********************************************/

static void calcBoundaryFacePrimitiveValue(Face &f, Cell *c, const Vector &d)
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

static void calcInternalFacePrimitiveValue(Face &f)
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
        f.U = {u_0, v_0, w_0};
    } 
    else
    {
        const Scalar u_1 = f.c1->U.x() + f.c1->grad_U.col(0).dot(f.r1);
        const Scalar v_1 = f.c1->U.y() + f.c1->grad_U.col(1).dot(f.r1);
        const Scalar w_1 = f.c1->U.z() + f.c1->grad_U.col(2).dot(f.r1);
        f.U = {u_1, v_1, w_1};
    }

    /// density
    if (f.rhoU.dot(f.n01) > 0)
        f.rho = f.c0->rho + f.c0->grad_rho.dot(f.r0);
    else
        f.rho = f.c1->rho + f.c1->grad_rho.dot(f.r1);
}

void calc_face_primitive_var()
{
    for (auto &f : face)
    {
        if (f.at_boundary)
        {
            if (f.c0)
                calcBoundaryFacePrimitiveValue(f, f.c0, f.r0);
            else if (f.c1)
                calcBoundaryFacePrimitiveValue(f, f.c1, f.r1);
            else
                throw empty_connectivity(f.index);
        } else
            calcInternalFacePrimitiveValue(f);
    }
}

void calc_face_viscous_shear_stress()
{
    for(auto &f : face)
    {
        if(f.at_boundary)
        {
            Cell *c;
            bool adj_to_0;
            if(f.c0)
            {
                c = f.c0;
                adj_to_0 = true;
            }
            else if(f.c1)
            {
                c = f.c1;
                adj_to_0 = false;
            }
            else
                throw empty_connectivity(f.index);

            const Vector &n = adj_to_0 ? f.n01 : f.n10;
            const Vector &r = adj_to_0 ? f.r0 : f.r1;

            auto p = f.parent;
            if(p->BC==BC_PHY::Wall)
            {
                Vector dU = c->U - f.U;
                dU -= dU.dot(n) * n;
                const Vector tw = -f.mu / r.dot(n) * dU;
                f.tau = tw * n.transpose();
            }
            else if(p->BC == BC_PHY::Symmetry)
            {
                Vector dU = c->U.dot(n) * n;
                const Vector t_cz = -2.0 * f.mu * dU / r.norm();
                f.tau = t_cz * n.transpose();
            }
            else if(p->BC == BC_PHY::Inlet || p->BC == BC_PHY::Outlet)
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
    for (auto &c : cell)
    {
        /// Dynamic viscosity
        // c.mu = Sutherland(c.T);
        c.mu = c.rho / Re;
    }

    /// Enforce boundary conditions for primitive variables.
    set_bc_of_primitive_var();

    calcFaceGhostVariable();

    /// Gradients of primitive variables at centroid of each cell.
    calc_cell_primitive_gradient();

    /// Gradients of primitive variables at centroid of each face.
    calc_face_primitive_gradient();

    /// Interpolate values of primitive variables on each face.
    calc_face_primitive_var();

    /// Update physical properties at centroid of each face.
    for (auto &f : face)
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

/*********************************************** Temporal Discretization **********************************************/

/**
 * Transient time-step for each explicit marching iteration.
 * @return Current time-step used for temporal integration.
 */
Scalar calcTimeStep()
{
    Scalar ret = 5e-3;
    return ret;
}

/**
 * 1st-order explicit time-marching.
 * Pressure-Velocity coupling is solved using Fractional-Step Method.
 * @param TimeStep
 */
void ForwardEuler(Scalar TimeStep)
{
    reconstruction();

    /// Count flux of each cell.
    calc_cell_flux();

    /// Prediction Step
    for (auto &c : cell)
        c.rhoU_star = c.rhoU + TimeStep / c.volume * (-c.convection_flux - c.pressure_flux + c.viscous_flux);

    /// rhoU* at internal face.
    for (auto &f : face)
    {
        if (!f.at_boundary)
        {
            f.rhoU_star = f.ksi0 * f.c0->rhoU_star + f.ksi1 * f.c1->rhoU_star;
            const Vector mean_grad_p = 0.5*(f.c1->grad_p + f.c0->grad_p);
            const Vector d10 = f.c1->centroid - f.c0->centroid;
            const Vector compact_grad_p = (f.c1->p - f.c0->p) / (d10.dot(d10)) * d10;
            const Vector rhoU_rc = -TimeStep * (compact_grad_p - mean_grad_p); /// Rhie-Chow interpolation
            f.rhoU_star += rhoU_rc;
        }
        f.grad_p_prime.setZero();
    }

    /// Correction Step
    Scalar res = 1.0;
    std::vector<Scalar> prev_dp(NumOfCell, 0.0);
    std::vector<Scalar> prev_dp_f(NumOfFace, 0.0);
    while(res > 1e-10)
    {
        res = 0.0;
        Scalar max_dp = 0.0, min_dp = 1e16;
        size_t max_dp_idx, min_dp_idx;

        calcPressureCorrectionEquationRHS(Q_dp_2, TimeStep);
        sx_solver_amg_solve(&dp_solver_2, &x_dp_2, &Q_dp_2);
        for (int i = 0; i < NumOfCell; ++i)
        {
            auto &c = cell.at(i);
            c.p_prime = sx_vec_get_entry(&x_dp_2, i);
            res += std::fabs(c.p_prime - prev_dp[i]);
            prev_dp[i] = c.p_prime;
            if(std::fabs(c.p_prime) > max_dp)
            {
                max_dp = std::fabs(c.p_prime);
                max_dp_idx = c.index;
            }
            if(std::fabs(c.p_prime) < min_dp)
            {
                min_dp = std::fabs(c.p_prime);
                min_dp_idx = c.index;
            }
        }
        calc_cell_pressure_correction_gradient();
        calc_face_pressure_correction_gradient();
        res /= NumOfCell;
        std::cout << "||dp - dp_prev|| = " << res << std::endl;
        std::cout << "|dp|_max = " << max_dp << " at " << max_dp_idx << std::endl;
        std::cout << "|dp|_min = " << min_dp << " at " << min_dp_idx << std::endl;
    }

    /// Update
    for (auto &f : face)
    {
        if(!f.at_boundary)
            f.rhoU = f.rhoU_star - TimeStep * f.grad_p_prime;
    }
    for (auto &c : cell)
    {
        /// Velocity
        c.rhoU = c.rhoU_star - TimeStep * c.grad_p_prime;
        c.U = c.rhoU / c.rho;

        /// Pressure
        c.p += c.p_prime;
    }
}

/********************************************************* END ********************************************************/
