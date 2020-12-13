#include <string>
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
            e.U_BC = Dirichlet;
            e.p_BC = Neumann;
            e.T_BC = Dirichlet;
            e.BC = BC_PHY::Wall;
        }
        else if (e.name == "DOWN")
        {
            e.U_BC = Dirichlet;
            e.p_BC = Neumann;
            e.T_BC = Dirichlet;
            e.BC = BC_PHY::Wall;
        }
        else if (e.name == "LEFT")
        {
            e.U_BC = Dirichlet;
            e.p_BC = Neumann;
            e.T_BC = Neumann;
            e.BC = BC_PHY::Wall;
        }
        else if (e.name == "RIGHT")
        {
            e.U_BC = Dirichlet;
            e.p_BC = Neumann;
            e.T_BC = Neumann;
            e.BC = BC_PHY::Wall;
        }
        else if (e.name == "FRONT")
        {
            e.U_BC = Dirichlet;
            e.p_BC = Neumann;
            e.T_BC = Neumann;
            e.BC = BC_PHY::Wall;
        }
        else if (e.name == "BACK")
        {
            e.U_BC = Dirichlet;
            e.p_BC = Neumann;
            e.T_BC = Neumann;
            e.BC = BC_PHY::Wall;
        }
        else
            throw unexpected_patch(e.name);
    }
}

static const Vector U_UP = { 1.0, 0.0, 0.0 }; // m/s
static const Scalar T_DOWN = 300.0, T_UP = 1500.0; // K

/**
 * Boundary conditions on all related faces for all variables.
 */
void BC_Primitive()
{
    for (const auto &e : patch)
    {
        if (e.name == "UP")
        {
            for (auto f : e.surface)
            {
                f->U = U_UP;
                f->sn_grad_p = ZERO_SCALAR;
                f->T = T_UP;
            }
        }
        else if (e.name == "DOWN")
        {
            for (auto f : e.surface)
            {
                f->U = ZERO_VECTOR;
                f->sn_grad_p = ZERO_SCALAR;
                f->T = T_DOWN;
            }
        }
        else if (e.name == "LEFT")
        {
            for (auto f : e.surface)
            {
                f->U = ZERO_VECTOR;
                f->sn_grad_p = ZERO_SCALAR;
                f->sn_grad_T = ZERO_SCALAR;
            }
        }
        else if (e.name == "RIGHT")
        {
            for (auto f : e.surface)
            {
                f->U = ZERO_VECTOR;
                f->sn_grad_p = ZERO_SCALAR;
                f->sn_grad_T = ZERO_SCALAR;
            }
        }
        else if (e.name == "FRONT")
        {
            for (auto f : e.surface)
            {
                f->U = ZERO_VECTOR;
                f->sn_grad_p = ZERO_SCALAR;
                f->sn_grad_T = ZERO_SCALAR;
            }
        }
        else if (e.name == "BACK")
        {
            for (auto f : e.surface)
            {
                f->U = ZERO_VECTOR;
                f->sn_grad_p = ZERO_SCALAR;
                f->sn_grad_T = ZERO_SCALAR;
            }
        }
        else
            throw unexpected_patch(e.name);
    }
}

void BC_Nodal()
{
    for(const auto &e : patch)
    {
        if (e.name == "UP")
        {
            for (auto v : e.vertex)
            {
                v->U = U_UP;
                v->T = T_UP;
            }
        }
        else if (e.name == "DOWN")
        {
            for (auto v : e.vertex)
            {
                v->U = ZERO_VECTOR;
                v->T = T_DOWN;
            }
        }
        else if (e.name == "LEFT")
        {
            for (auto v : e.vertex)
            {
                v->U = ZERO_VECTOR;
            }
        }
        else if (e.name == "RIGHT")
        {
            for (auto v : e.vertex)
            {
                v->U = ZERO_VECTOR;
            }
        }
        else if (e.name == "FRONT")
        {
            for (auto v : e.vertex)
            {
                v->U = ZERO_VECTOR;
            }
        }
        else if (e.name == "BACK")
        {
            for (auto v : e.vertex)
            {
                v->U = ZERO_VECTOR;
            }
        }
        else
            throw unexpected_patch(e.name);
    }
}
