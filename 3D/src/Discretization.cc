#include "../inc/BC.h"
#include "../inc/PoissonEqn.h"
#include "../inc/CHEM.h"
#include "../inc/IC.h"
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

void calcCellProperty()
{
    for (auto &c : cell)
    {
        /// Dynamic viscosity
        // c.mu = Sutherland(c.T);
        c.mu = c.rho / Re;
    }
}

void calcFaceProperty()
{
    for (auto &f : face)
    {
        /// Dynamic viscosity
        // f.mu = Sutherland(f.T);
        f.mu = f.rho / Re;
    }
}

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
    if (f.U.dot(f.n01) > 0)
    {
        const Scalar u_0 = f.c0->U.x() + f.c0->grad_U.col(0).dot(f.r0);
        const Scalar v_0 = f.c0->U.y() + f.c0->grad_U.col(1).dot(f.r0);
        const Scalar w_0 = f.c0->U.z() + f.c0->grad_U.col(2).dot(f.r0);
        f.U = {u_0, v_0, w_0};
    } else
    {
        const Scalar u_1 = f.c1->U.x() + f.c1->grad_U.col(0).dot(f.r1);
        const Scalar v_1 = f.c1->U.y() + f.c1->grad_U.col(1).dot(f.r1);
        const Scalar w_1 = f.c1->U.z() + f.c1->grad_U.col(2).dot(f.r1);
        f.U = {u_1, v_1, w_1};
    }

    /// density
    if (f.U.dot(f.n01) > 0)
        f.rho = f.c0->rho + f.c0->grad_rho.dot(f.r0);
    else
        f.rho = f.c1->rho + f.c1->grad_rho.dot(f.r1);
}

void calcFaceValue()
{
    for (auto &f : face)
    {
        /// Primitive variables.
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

        /// Conservative variables.
        // f.rhoU = f.rho * f.U;
    }
}

void calcFaceViscousStress()
{
    for (auto &f : face)
    {
        const Scalar loc_div3 = (f.grad_U(0, 0) + f.grad_U(1, 1) + f.grad_U(2, 2)) / 3.0;

        f.tau(0, 0) = 2 * f.mu * (f.grad_U(0, 0) - loc_div3);
        f.tau(1, 1) = 2 * f.mu * (f.grad_U(1, 1) - loc_div3);
        f.tau(2, 2) = 2 * f.mu * (f.grad_U(2, 2) - loc_div3);

        f.tau(0, 1) = f.tau(1, 0) = f.mu * (f.grad_U(0, 1) + f.grad_U(1, 0));
        f.tau(1, 2) = f.tau(2, 1) = f.mu * (f.grad_U(1, 2) + f.grad_U(2, 1));
        f.tau(2, 0) = f.tau(0, 2) = f.mu * (f.grad_U(2, 0) + f.grad_U(0, 2));
    }
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
    /// Init
    for (size_t i = 1; i <= NumOfCell; ++i)
    {
        auto &c = cell(i);
        c.rho = c.rho0;
        c.U = c.U0;
        c.p = c.p0;
        c.T = c.T0;
    }

    /// Enforce boundary conditions.
    BC();

    /// Update physical properties at centroid of each cell.
    calcCellProperty();

    /// Gradients at centroid of each cell.
    calc_cell_primitive_gradient();

    /// Gradients at centroid of each face.
    calc_face_primitive_gradient();

    /// Interpolate values on each face.
    calcFaceValue();

    /// Update physical properties at centroid of each face.
    calcFaceProperty();

    /// Viscous stress on each face.
    calcFaceViscousStress();

    /// Count flux of each cell.
    calcCellFlux();

    /// Prediction Step
    for (auto &c : cell)
        c.rhoU_star = c.rhoU0 + TimeStep / c.volume * (-c.convection_flux - c.pressure_flux + c.viscous_flux);

    /// rhoU* at internal face.
    for (auto &f : face)
    {
        if (!f.atBdry)
            f.rhoU_star = f.ksi0 * f.c0->rhoU_star + f.ksi1 * f.c1->rhoU_star;
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

    for (auto &f : face)
        f.rhoU = f.rhoU_star - TimeStep * f.grad_p_prime;

    /// Update
    for (int i = 0; i < NumOfCell; ++i)
    {
        auto &c = cell.at(i);

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
}

/********************************************************* END ********************************************************/
