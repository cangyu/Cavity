#include "../inc/CHEM.h"
#include "../inc/BC.h"
#include "../inc/Spatial.h"

extern int NumOfPnt, NumOfFace, NumOfCell;
extern NaturalArray<Point> pnt;
extern NaturalArray<Face> face;
extern NaturalArray<Cell> cell;
extern NaturalArray<Patch> patch;

void interp_nodal_primitive_var()
{
    for (int i = 1; i <= NumOfPnt; ++i)
    {
        auto &n_dst = pnt(i);

        const auto &dc = n_dst.dependent_cell;
        const auto &wf = n_dst.cell_weights;

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
            n_dst.rho += cwf * dc.at(j)->rho;
            n_dst.U += cwf * dc.at(j)->U;
            n_dst.p += cwf * dc.at(j)->p;
            n_dst.T += cwf * dc.at(j)->T;
        }
    }

    set_bc_nodal();
}

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

void calc_cell_viscous_shear_stress_next()
{
    for(auto &c : cell)
    {
        Stokes(c.viscosity, c.grad_U_next, c.tau_next);
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
                Vector dU = c->U_next - f.U;
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
