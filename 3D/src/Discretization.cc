#include "../inc/BC.h"
#include "../inc/PoissonEqn.h"
#include "../inc/CHEM.h"
#include "../inc/Gradient.h"
#include "../inc/Flux.h"
#include "../inc/Discretization.h"

extern int NumOfPnt;
extern int NumOfFace;
extern int NumOfCell;
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

/************************************************ Physical Property ***************************************************/

void update_cell_property()
{
    for (auto &c : cell)
    {
        /// Dynamic viscosity
        // c.mu = Sutherland(c.T);
        c.mu = c.rho / Re;
    }
}

void update_face_property()
{
    for (auto &f : face)
    {
        /// Dynamic viscosity
        // f.mu = Sutherland(f.T);
        f.mu = f.rho / Re;
    }
}

/*********************************************** Spatial Discretization ***********************************************/

void interp_nodal_primitive_var()
{
    for (int i = 1; i <= NumOfPnt; ++i)
    {
        auto &n_dst = pnt(i);

        const auto &dc = n_dst.dependentCell;
        const auto &wf = n_dst.cellWeightingCoef;

        const auto N = dc.size();
        if (N != wf.size())
            throw std::runtime_error("Inconsistency detected!");

        n_dst.rho = ZERO_SCALAR;
        n_dst.U = ZERO_VECTOR;
        n_dst.p = ZERO_SCALAR;
        n_dst.T = ZERO_SCALAR;
        for (int j = 0; j < N; ++j)
        {
            const auto cwf = wf.at(j);
            n_dst.rho += cwf * dc.at(j)->rho0;
            n_dst.U += cwf * dc.at(j)->U0;
            n_dst.p += cwf * dc.at(j)->p0;
            n_dst.T += cwf * dc.at(j)->T0;
        }
    }
}

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
        if (f.atBdry)
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
        if(f.atBdry)
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
            if(bc_is_wall(p->BC))
            {
                Vector dU = c->U - f.U;
                dU -= dU.dot(n) * n;
                const Vector tw = -f.mu / r.dot(n) * dU;
                f.tau = tw * n.transpose();
            }
            else if(bc_is_symmetry(p->BC))
            {
                Vector dU = c->U.dot(n) * n;
                const Vector t_cz = -2.0 * f.mu * dU / r.norm();
                f.tau = t_cz * n.transpose();
            }
            else if(bc_is_inlet(p->BC) || bc_is_outlet(p->BC))
                Stokes(f.mu, f.grad_U, f.tau);
            else
                throw unsupported_boundary_condition(get_bc_name(p->BC));
        }
        else
            Stokes(f.mu, f.grad_U, f.tau);
    }
}

void prepare_next_run()
{
    /// Init primitive variables
    for (auto &c : cell)
    {
        c.rho = c.rho0;
        c.U = c.U0;
        c.p = c.p0;
        c.T = c.T0;
    }

    /// Update physical properties at centroid of each cell.
    update_cell_property();

    /// Enforce boundary conditions for primitive variables.
    set_bc_of_primitive_var();

    /// Gradients of primitive variables at centroid of each cell.
    calc_cell_primitive_gradient();

    /// Gradients of primitive variables at centroid of each face.
    calc_face_primitive_gradient();

    /// Interpolate values of primitive variables on each face.
    calc_face_primitive_var();

    /// Update physical properties at centroid of each face.
    update_face_property();

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
    /// Count flux of each cell.
    calc_cell_flux();

    /// Prediction Step
    for (auto &c : cell)
        c.rhoU_star = c.rhoU0 + TimeStep / c.volume * (-c.convection_flux - c.pressure_flux + c.viscous_flux);

    /// rhoU* at internal face.
    for (auto &f : face)
    {
        if (!f.atBdry)
        {
            f.rhoU_star = f.ksi0 * f.c0->rhoU_star + f.ksi1 * f.c1->rhoU_star - TimeStep * (f.grad_p - 0.5*(f.c0->grad_p + f.c1->grad_p));
        }
    }

    /// Correction Step
    for (int k = 0; k < NOC_ITER; ++k)
    {
        calcPressureCorrectionEquationRHS(Q_dp_2, TimeStep);
        sx_solver_amg_solve(&dp_solver_2, &x_dp_2, &Q_dp_2);
        for (int i = 0; i < NumOfCell; ++i)
        {
            auto &c = cell.at(i);
            c.p_prime = sx_vec_get_entry(&x_dp_2, i);
        }
        calc_cell_pressure_correction_gradient();
        calc_face_pressure_correction_gradient();
    }

    /// Update
    for (auto &f : face)
    {
        if(!f.atBdry)
            f.rhoU = f.rhoU_star - TimeStep * f.grad_p_prime;
    }
    for (auto &c : cell)
    {
        /// Density
        c.continuity_res = 0.0;
        c.rho0 = c.rho + TimeStep * c.continuity_res;

        /// Velocity
        c.rhoU0 = c.rhoU_star - TimeStep * c.grad_p_prime;
        c.U0 = c.rhoU0 / c.rho0;

        /// Pressure
        c.p0 = c.p + c.p_prime;

        /// Temperature
        c.energy_res = 0.0;
        c.T0 = c.T + TimeStep * c.energy_res;
    }

    prepare_next_run();
}

/********************************************************* END ********************************************************/
