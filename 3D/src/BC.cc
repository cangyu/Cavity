#include <string>
#include "../inc/custom_type.h"
#include "../inc/BC.h"

extern int NumOfPnt, NumOfFace, NumOfCell;
extern NaturalArray<Point> pnt;
extern NaturalArray<Face> face;
extern NaturalArray<Cell> cell;
extern NaturalArray<Patch> patch;

void BC_TABLE()
{
    for (auto &e : patch)
    {
        if (e.name == "UP")
        {
            e.rho_BC = Dirichlet;
            e.U_BC[0] = Dirichlet;
            e.U_BC[1] = Dirichlet;
            e.U_BC[2] = Dirichlet;
            e.p_BC = Neumann;
            e.p_prime_BC = Neumann;
            e.T_BC = Dirichlet;
            e.BC = BC_PHY::Wall;
        }
        else if (e.name == "DOWN")
        {
            e.rho_BC = Dirichlet;
            e.U_BC[0] = Dirichlet;
            e.U_BC[1] = Dirichlet;
            e.U_BC[2] = Dirichlet;
            e.p_BC = Neumann;
            e.p_prime_BC = Neumann;
            e.T_BC = Dirichlet;
            e.BC = BC_PHY::Wall;
        }
        else if (e.name == "LEFT")
        {
            e.rho_BC = Dirichlet;
            e.U_BC[0] = Dirichlet;
            e.U_BC[1] = Dirichlet;
            e.U_BC[2] = Dirichlet;
            e.p_BC = Neumann;
            e.p_prime_BC = Neumann;
            e.T_BC = Neumann;
            e.BC = BC_PHY::Wall;
        }
        else if (e.name == "RIGHT")
        {
            e.rho_BC = Dirichlet;
            e.U_BC[0] = Dirichlet;
            e.U_BC[1] = Dirichlet;
            e.U_BC[2] = Dirichlet;
            e.p_BC = Neumann;
            e.p_prime_BC = Neumann;
            e.T_BC = Neumann;
            e.BC = BC_PHY::Wall;
        }
        else if (e.name == "FRONT")
        {
            e.rho_BC = Dirichlet;
            e.U_BC[0] = Dirichlet;
            e.U_BC[1] = Dirichlet;
            e.U_BC[2] = Dirichlet;
            e.p_BC = Neumann;
            e.p_prime_BC = Neumann;
            e.T_BC = Neumann;
            e.BC = BC_PHY::Wall;
        }
        else if (e.name == "BACK")
        {
            e.rho_BC = Dirichlet;
            e.U_BC[0] = Dirichlet;
            e.U_BC[1] = Dirichlet;
            e.U_BC[2] = Dirichlet;
            e.p_BC = Neumann;
            e.p_prime_BC = Neumann;
            e.T_BC = Neumann;
            e.BC = BC_PHY::Wall;
        }
        else
            throw unexpected_patch(e.name);
    }
}

static const Scalar rho0 = 1.225; // kg/m3
static const Vector U_UP = { 1.0, 0.0, 0.0 }; // m/s
static const Scalar T_DOWN = 300.0, T_UP = 1500.0; // K

/**
 * Boundary conditions on all related faces for all variables.
 */
void set_bc_of_primitive_var()
{
    for (const auto &e : patch)
    {
        if (e.name == "UP")
        {
            for (auto f : e.surface)
            {
                f->rho = rho0;
                f->U = U_UP;
                f->sn_grad_p = ZERO_SCALAR;
                f->T = T_UP;
            }
        }
        else if (e.name == "DOWN")
        {
            for (auto f : e.surface)
            {
                f->rho = rho0;
                f->U = ZERO_VECTOR;
                f->sn_grad_p = ZERO_SCALAR;
                f->T = T_DOWN;
            }
        }
        else if (e.name == "LEFT")
        {
            for (auto f : e.surface)
            {
                f->rho = rho0;
                f->U = ZERO_VECTOR;
                f->sn_grad_p = ZERO_SCALAR;
                f->sn_grad_T = ZERO_SCALAR;
            }
        }
        else if (e.name == "RIGHT")
        {
            for (auto f : e.surface)
            {
                f->rho = rho0;
                f->U = ZERO_VECTOR;
                f->sn_grad_p = ZERO_SCALAR;
                f->sn_grad_T = ZERO_SCALAR;
            }
        }
        else if (e.name == "FRONT")
        {
            for (auto f : e.surface)
            {
                f->rho = rho0;
                f->U = ZERO_VECTOR;
                f->sn_grad_p = ZERO_SCALAR;
                f->sn_grad_T = ZERO_SCALAR;
            }
        }
        else if (e.name == "BACK")
        {
            for (auto f : e.surface)
            {
                f->rho = rho0;
                f->U = ZERO_VECTOR;
                f->sn_grad_p = ZERO_SCALAR;
                f->sn_grad_T = ZERO_SCALAR;
            }
        }
        else
            throw unexpected_patch(e.name);
    }
}

void set_bc_of_pressure_correction()
{
    for (const auto &e : patch)
        for (auto f : e.surface)
            f->sn_grad_p_prime.setZero();
}

void set_bc_nodal()
{
    for(const auto &e : patch)
    {
        if (e.name == "UP")
        {
            for (auto f : e.vertex)
            {
                f->rho = rho0;
                f->U = U_UP;
                f->T = T_UP;
            }
        }
        else if (e.name == "DOWN")
        {
            for (auto f : e.vertex)
            {
                f->rho = rho0;
                f->U = ZERO_VECTOR;
                f->T = T_DOWN;
            }
        }
        else if (e.name == "LEFT")
        {
            for (auto f : e.vertex)
            {
                f->rho = rho0;
                f->U = ZERO_VECTOR;
            }
        }
        else if (e.name == "RIGHT")
        {
            for (auto f : e.vertex)
            {
                f->rho = rho0;
                f->U = ZERO_VECTOR;
            }
        }
        else if (e.name == "FRONT")
        {
            for (auto f : e.vertex)
            {
                f->rho = rho0;
                f->U = ZERO_VECTOR;
            }
        }
        else if (e.name == "BACK")
        {
            for (auto f : e.vertex)
            {
                f->rho = rho0;
                f->U = ZERO_VECTOR;
            }
        }
        else
            throw unexpected_patch(e.name);
    }
}
