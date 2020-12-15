#include "../inc/BC.h"
#include "../inc/CHEM.h"
#include "../inc/Spatial.h"

extern int NumOfPnt, NumOfFace, NumOfCell;
extern NaturalArray<Point> pnt;
extern NaturalArray<Face> face;
extern NaturalArray<Cell> cell;
extern NaturalArray<Patch> patch;

void INTERP_Face_Temperature_star()
{
    for (auto& f : face)
    {
        if (f.at_boundary)
        {
            const auto T_BC = f.parent->T_BC;
            if (T_BC == Dirichlet)
                f.T_star = f.T;
            else if (T_BC == Neumann)
            {
                auto c = f.c0 ? f.c0 : f.c1;
                const Vector &r = f.c0 ? f.r0 : f.r1;
                f.T_star = c->T_star + f.grad_T.dot(r);
            }
            else
                throw unsupported_boundary_condition(T_BC);
        }
        else
            f.T_star = f.ksi0 * f.c0->T_star + f.ksi1 * f.c1->T_star; /// Less accurate, but bounded
    }
}

void INTERP_Face_Temperature_next()
{
    for (auto& f : face)
    {
        if (f.at_boundary)
        {
            const auto T_BC = f.parent->T_BC;
            if (T_BC == Dirichlet)
                f.T_next = f.T;
            else if (T_BC == Neumann)
            {
                auto c = f.c0 ? f.c0 : f.c1;
                const Vector &r = f.c0 ? f.r0 : f.r1;
                f.T_next = c->T_next + f.grad_T.dot(r);
            }
            else
                throw unsupported_boundary_condition(T_BC);
        }
        else
            f.T_next = f.ksi0 * f.c0->T_next + f.ksi1 * f.c1->T_next; /// Less accurate, but bounded
    }
}

void INTERP_Face_MassFlux_star(Scalar TimeStep)
{
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

void INTERP_Face_Velocity_next()
{
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
                f.U_next = C->U_star + r.transpose() * f.grad_U;
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
}

void INTERP_Face_Pressure_next()
{
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
                f.p_next = C->p_next + f.grad_p.dot(d);
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
}

void INTERP_Face_snGrad_PressureCorrection()
{
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
}

void RECONST_Cell_Grad_PressureCorrection()
{
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

void INTERP_Face_Primitive()
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

void CALC_Cell_ViscousShearStress_next()
{
    for (auto &C : cell)
    {
        Stokes(C.viscosity, C.grad_U_next, C.tau_next);
    }
}

void CALC_Face_ViscousShearStress_next()
{
    for (auto& f : face)
    {
        if (f.at_boundary)
        {
            auto c = f.c0 ? f.c0 : f.c1;
            const Vector& n = f.c0 ? f.n01 : f.n10;
            const Vector& r = f.c0 ? f.r0 : f.r1;

            const auto BC = f.parent->BC;
            if (BC == BC_PHY::Wall)
            {
                Vector dU = c->U_next - f.U;
                dU -= dU.dot(n) * n;
                const Vector tw = -f.viscosity / r.dot(n) * dU;
                f.tau_next = tw * n.transpose();
            }
            else if (BC == BC_PHY::Symmetry)
            {
                Vector dU = c->U_next.dot(n) * n;
                const Vector t_cz = -2.0 * f.viscosity * dU / r.norm();
                f.tau_next = t_cz * n.transpose();
            }
            else if (BC == BC_PHY::Inlet || BC == BC_PHY::Outlet)
                Stokes(f.viscosity, f.grad_U_next, f.tau_next);
            else
                throw unsupported_boundary_condition(BC);
        }
        else
            Stokes(f.viscosity, f.grad_U_next, f.tau_next);
    }
}

void CALC_Face_ViscousShearStress()
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

void INTERP_Node_Primitive()
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

    BC_Nodal();
}

void prepare_first_run()
{
    BC_Primitive();

}
