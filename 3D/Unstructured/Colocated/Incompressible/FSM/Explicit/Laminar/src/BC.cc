#include "../inc/custom_type.h"
#include "../inc/BC.h"

extern size_t NumOfPnt, NumOfFace, NumOfCell;
extern NaturalArray<Point> pnt;
extern NaturalArray<Face> face;
extern NaturalArray<Cell> cell;
extern NaturalArray<Patch> patch;

void BC_TABLE()
{
    for (const auto &e : patch)
    {
        if (e.name == "UP")
        {
            for (auto f : e.surface)
            {
                f->rho_BC = Dirichlet;
                f->U_BC = { Dirichlet, Dirichlet, Dirichlet };
                f->p_BC = Neumann;
                f->p_prime_BC = Neumann;
                f->T_BC = Dirichlet;
            }
        }
        else if (e.name == "DOWN")
        {
            for (auto f : e.surface)
            {
                f->rho_BC = Dirichlet;
                f->U_BC = { Dirichlet, Dirichlet, Dirichlet };
                f->p_BC = Neumann;
                f->p_prime_BC = Neumann;
                f->T_BC = Dirichlet;
            }
        }
        else if (e.name == "LEFT")
        {
            for (auto f : e.surface)
            {
                f->rho_BC = Dirichlet;
                f->U_BC = { Dirichlet, Dirichlet, Dirichlet };
                f->p_BC = Neumann;
                f->p_prime_BC = Neumann;
                f->T_BC = Neumann;
            }
        }
        else if (e.name == "RIGHT")
        {
            for (auto f : e.surface)
            {
                f->rho_BC = Dirichlet;
                f->U_BC = { Dirichlet, Dirichlet, Dirichlet };
                f->p_BC = Neumann;
                f->p_prime_BC = Neumann;
                f->T_BC = Neumann;
            }
        }
        else if (e.name == "FRONT")
        {
            for (auto f : e.surface)
            {
                f->rho_BC = Dirichlet;
                f->U_BC = { Dirichlet, Dirichlet, Dirichlet };
                f->p_BC = Neumann;
                f->p_prime_BC = Neumann;
                f->T_BC = Neumann;
            }
        }
        else if (e.name == "BACK")
        {
            for (auto f : e.surface)
            {
                f->rho_BC = Dirichlet;
                f->U_BC = { Dirichlet, Dirichlet, Dirichlet };
                f->p_BC = Neumann;
                f->p_prime_BC = Neumann;
                f->T_BC = Neumann;
            }
        }
        else
            throw unexpected_patch(e.name);
    }
}

static const Scalar rho0 = 1.225; // kg/m3
static const Vector U_UP = { 1e-3, 0.0, 0.0 }; // m/s
static const Scalar T_DOWN = 300.0, T_UP = 300.0; // K

/**
 * Boundary conditions on all related faces for all variables.
 */
void BC()
{
    for (const auto &e : patch)
    {
        if (e.name == "UP")
        {
            for (auto f : e.surface)
            {
                f->rho = rho0;
                f->U = U_UP;
                f->grad_p_prime = ZERO_VECTOR;
                f->T = T_UP;
            }
        }
        else if (e.name == "DOWN")
        {
            for (auto f : e.surface)
            {
                f->rho = rho0;
                f->U = ZERO_VECTOR;
                f->grad_p_prime = ZERO_VECTOR;
                f->T = T_DOWN;
            }
        }
        else if (e.name == "LEFT")
        {
            for (auto f : e.surface)
            {
                f->rho = rho0;
                f->U = ZERO_VECTOR;
                f->grad_p_prime = ZERO_VECTOR;
                f->grad_T = ZERO_VECTOR;
            }
        }
        else if (e.name == "RIGHT")
        {
            for (auto f : e.surface)
            {
                f->rho = rho0;
                f->U = ZERO_VECTOR;
                f->grad_p_prime = ZERO_VECTOR;
                f->grad_T = ZERO_VECTOR;
            }
        }
        else if (e.name == "FRONT")
        {
            for (auto f : e.surface)
            {
                f->rho = rho0;
                f->U = ZERO_VECTOR;
                f->grad_p_prime = ZERO_VECTOR;
                f->grad_T = ZERO_VECTOR;
            }
        }
        else if (e.name == "BACK")
        {
            for (auto f : e.surface)
            {
                f->rho = rho0;
                f->U = ZERO_VECTOR;
                f->grad_p_prime = ZERO_VECTOR;
                f->grad_T = ZERO_VECTOR;
            }
        }
        else
            throw unexpected_patch(e.name);
    }
}

void updateNodalValue()
{
    /* Interpolation */
    for (auto &n : pnt)
    {
        n.rho = ZERO_SCALAR;
        n.U = ZERO_VECTOR;
        n.p = ZERO_SCALAR;
        n.T = ZERO_SCALAR;
        for (int j = 1; j <= n.cellWeightingCoef.size(); ++j)
        {
            const auto curCoef = n.cellWeightingCoef(j);
            const auto curCell = n.dependentCell(j);

            n.rho += curCoef * curCell->rho0;
            n.U += curCoef * curCell->U0;
            n.p += curCoef * curCell->p0;
            n.T += curCoef * curCell->T0;
        }
    }

    std::vector<bool> visited(NumOfPnt + 1, false);

    /* Velocity */
    for (auto &e : patch)
    {
        if (e.name == "UP")
        {
            for (auto f : e.surface)
                for (auto v : f->vertex)
                    if (!visited[v->index])
                    {
                        v->U = U_UP;
                        visited[v->index] = true;
                    }
        }
    }
    for (auto & e : patch)
    {
        if (e.name != "UP")
        {
            for (auto f : e.surface)
                for (auto v : f->vertex)
                    if (!visited[v->index])
                    {
                        v->U = ZERO_VECTOR;
                        visited[v->index] = true;
                    }
        }
    }

    /* Temperature */
    std::fill(visited.begin(), visited.end(), false);
    for (auto &e : patch)
    {
        if (e.name == "UP")
        {
            for (auto f : e.surface)
                for (auto v : f->vertex)
                    if (!visited[v->index])
                    {
                        v->T = T_UP;
                        visited[v->index] = true;
                    }
        }
    }
    for (auto &e : patch)
    {
        if (e.name == "DOWN")
        {
            for (auto f : e.surface)
                for (auto v : f->vertex)
                    if (!visited[v->index])
                    {
                        v->T = T_DOWN;
                        visited[v->index] = true;
                    }
        }
    }
}
