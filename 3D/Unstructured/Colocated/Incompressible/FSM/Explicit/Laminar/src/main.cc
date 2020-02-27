#include <iostream>
#include <fstream>
#include <map>
#include <cmath>
#include <functional>
#include "../inc/custom_type.h"
#include "../inc/CHEM.h"
#include "../inc/IO.h"
#include "../inc/BC.h"
#include "../inc/IC.h"
#include "../inc/LeastSquare.h"
#include "../inc/PoissonEqn.h"
#include "../inc/Gradient.h"
#include "../inc/Flux.h"

/* Grid utilities */
size_t NumOfPnt = 0;
size_t NumOfFace = 0;
size_t NumOfCell = 0;

NaturalArray<Point> pnt; // Node objects
NaturalArray<Face> face; // Face objects
NaturalArray<Cell> cell; // Cell objects
NaturalArray<Patch> patch; // Group of boundary faces

/* Pressure-Corrrection equation coefficients */
static Eigen::SparseMatrix<Scalar> A_dp;
static Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Q_dp;
static Eigen::BiCGSTAB<Eigen::SparseMatrix<Scalar>, Eigen::IncompleteLUT<Scalar, int>> dp_solver;

/****************************************************** Property *****************************************************/

void calcCellProperty()
{
    for (auto &c : cell)
    {
        // Dynamic viscosity
        // c.mu = Sutherland(c.T);
        c.mu = 1.225 * 1e-2;
    }
}

void calcFaceProperty()
{
    for (auto &f : face)
    {
        // Dynamic viscosity
        // f.mu = Sutherland(f.T);
        f.mu = 1.225 * 1e-2;
    }
}

/*********************************************** Spatial Discretization **********************************************/

void calcFaceValue()
{
    for (auto &f : face)
    {
        if (f.atBdry)
        {
            if (f.c0)
            {
                // density
                switch (f.rho_BC)
                {
                case Dirichlet:
                    break;
                case Neumann:
                    f.rho = f.c0->rho + (f.grad_rho + f.c0->grad_rho).dot(f.r0) / 2;
                    break;
                case Robin:
                    throw unsupported_boundary_condition(Robin);
                default:
                    break;
                }

                // velocity-x
                switch (f.U_BC[0])
                {
                case Dirichlet:
                    break;
                case Neumann:
                    f.U.x() = f.c0->U.x() + (f.grad_U.col(0) + f.c0->grad_U.col(0)).dot(f.r0) / 2;
                    break;
                case Robin:
                    throw unsupported_boundary_condition(Robin);
                default:
                    break;
                }

                // velocity-y
                switch (f.U_BC[1])
                {
                case Dirichlet:
                    break;
                case Neumann:
                    f.U.y() = f.c0->U.y() + (f.grad_U.col(1) + f.c0->grad_U.col(1)).dot(f.r0) / 2;
                    break;
                case Robin:
                    throw unsupported_boundary_condition(Robin);
                default:
                    break;
                }

                // velocity-z
                switch (f.U_BC[2])
                {
                case Dirichlet:
                    break;
                case Neumann:
                    f.U.z() = f.c0->U.z() + (f.grad_U.col(2) + f.c0->grad_U.col(2)).dot(f.r0) / 2;
                    break;
                case Robin:
                    throw unsupported_boundary_condition(Robin);
                default:
                    break;
                }

                // pressure
                switch (f.p_BC)
                {
                case Dirichlet:
                    break;
                case Neumann:
                    f.p = f.c0->p + (f.grad_p + f.c0->grad_p).dot(f.r0) / 2;
                    break;
                case Robin:
                    throw unsupported_boundary_condition(Robin);
                default:
                    break;
                }

                // temperature
                switch (f.T_BC)
                {
                case Dirichlet:
                    break;
                case Neumann:
                    f.T = f.c0->T + (f.grad_T + f.c0->grad_T).dot(f.r0) / 2;
                    break;
                case Robin:
                    throw unsupported_boundary_condition(Robin);
                default:
                    break;
                }
            }
            else if (f.c1)
            {
                // density
                switch (f.rho_BC)
                {
                case Dirichlet:
                    break;
                case Neumann:
                    f.rho = f.c1->rho + (f.grad_rho + f.c1->grad_rho).dot(f.r1) / 2;
                    break;
                case Robin:
                    throw unsupported_boundary_condition(Robin);
                default:
                    break;
                }

                // velocity-x
                switch (f.U_BC[0])
                {
                case Dirichlet:
                    break;
                case Neumann:
                    f.U.x() = f.c1->U.x() + (f.grad_U.col(0) + f.c1->grad_U.col(0)).dot(f.r1) / 2;
                    break;
                case Robin:
                    throw unsupported_boundary_condition(Robin);
                default:
                    break;
                }

                // velocity-y
                switch (f.U_BC[1])
                {
                case Dirichlet:
                    break;
                case Neumann:
                    f.U.y() = f.c1->U.y() + (f.grad_U.col(1) + f.c1->grad_U.col(1)).dot(f.r1) / 2;
                    break;
                case Robin:
                    throw unsupported_boundary_condition(Robin);
                default:
                    break;
                }

                // velocity-z
                switch (f.U_BC[2])
                {
                case Dirichlet:
                    break;
                case Neumann:
                    f.U.z() = f.c1->U.z() + (f.grad_U.col(2) + f.c1->grad_U.col(2)).dot(f.r1) / 2;
                    break;
                case Robin:
                    throw unsupported_boundary_condition(Robin);
                default:
                    break;
                }

                // pressure
                switch (f.p_BC)
                {
                case Dirichlet:
                    break;
                case Neumann:
                    f.p = f.c1->p + (f.grad_p + f.c1->grad_p).dot(f.r1) / 2;
                    break;
                case Robin:
                    throw unsupported_boundary_condition(Robin);
                default:
                    break;
                }

                // temperature
                switch (f.T_BC)
                {
                case Dirichlet:
                    break;
                case Neumann:
                    f.T = f.c1->T + (f.grad_T + f.c1->grad_T).dot(f.r1) / 2;
                    break;
                case Robin:
                    throw unsupported_boundary_condition(Robin);
                default:
                    break;
                }
            }
            else
                throw empty_connectivity(f.index);
        }
        else
        {
            // weighting coefficient
            const Scalar ksi = f.r1.norm() / (f.r0.norm() + f.r1.norm());

            // pressure
            const Scalar p_0 = f.c0->p + f.c0->grad_p.dot(f.r0);
            const Scalar p_1 = f.c1->p + f.c1->grad_p.dot(f.r1);
            f.p = 0.5 * (p_0 + p_1);

            // temperature
            const Scalar T_0 = f.c0->T + f.c0->grad_T.dot(f.r0);
            const Scalar T_1 = f.c1->T + f.c1->grad_T.dot(f.r1);
            f.T = ksi * T_0 + (1.0 - ksi) * T_1;

            // velocity
            if (f.U.dot(f.n01) > 0)
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

            // density
            if (f.U.dot(f.n01) > 0)
                f.rho = f.c0->rho + f.c0->grad_rho.dot(f.r0);
            else
                f.rho = f.c1->rho + f.c1->grad_rho.dot(f.r1);
        }
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

void calcInternalFace_rhoU_star()
{
    for (auto &f : face)
    {
        if (!f.atBdry)
        {
            f.rhoU_star = (f.c0->rhoU_star + f.c1->rhoU0) / 2;
        }
    }
}

/*********************************************** Temporal Discretization *********************************************/

/**
 * Explicit Fractional-Step Method.
 * @param TimeStep
 */
void FSM(Scalar TimeStep)
{
    BC();

    // Physical properties at centroid of each cell
    calcCellProperty();

    // Boundary ghost values if any
    calcFaceGhostVariable();

    // Gradients at centroid of each cell
    calcCellGradient();

    // Gradients at centroid of each face
    calcFaceGradient();

    // Interpolate values on each face
    calcFaceValue();

    // Physical properties at centroid of each face
    calcFaceProperty();

    calcFaceViscousStress();

    calcCellFlux();

    // Prediction
    for (auto &c : cell)
        c.rhoU_star = c.rhoU0 + TimeStep / c.volume * (-c.convection_flux - c.pressure_flux + c.viscous_flux);

    // rhoU_star at each face
    calcInternalFace_rhoU_star();

    // Correction
    calcPressureCorrectionEquationRHS(Q_dp);
    Q_dp /= TimeStep;
    Eigen::VectorXd dp = dp_solver.solve(Q_dp);
    for (int i = 0; i < NumOfCell; ++i)
    {
        auto &c = cell.at(i);
        c.p_prime = dp(i);
    }
    calcPressureCorrectionGradient();

    // Update
    for (int i = 0; i < NumOfCell; ++i)
    {
        auto &c = cell.at(i);

        c.p += c.p_prime;
        c.U = (c.rhoU_star - TimeStep * c.grad_p_prime) / c.rho;

        c.continuity_res = 0.0;
        c.energy_res = 0.0;
    }
}

/**
 * 1st-order explicit time-marching.
 * @param TimeStep
 */
void ForwardEuler(Scalar TimeStep)
{
    /* Init */
    for (size_t i = 1; i <= NumOfCell; ++i)
    {
        auto &c = cell(i);
        c.rho = c.rho0;
        c.U = c.U0;
        c.p = c.p0;
        c.T = c.T0;
    }

    /* Pressure-Velocity coupling */
    FSM(TimeStep);

    /* Update */
    for (size_t i = 1; i <= NumOfCell; ++i)
    {
        auto &c = cell(i);

        c.rho = c.rho0 + TimeStep * c.continuity_res;
        c.T = c.T0 + TimeStep * c.energy_res;

        c.rho0 = c.rho;
        c.U0 = c.U;
        c.p0 = c.p;
        c.T0 = c.T;
        c.rhoU0 = c.rho0 * c.U0;
    }
}

/***************************************************** Solution Control **********************************************/

/* Iteration timing and counting */
const int MAX_ITER = 2000;
const Scalar MAX_TIME = 100.0; // s

static const size_t OUTPUT_GAP = 2;
static const std::string SEP = "  ";

std::ostream &LOG_OUT = std::cout;

static void stat_min_max(const std::string& var_name, std::function<Scalar(const Cell&)> extractor)
{
    Scalar var_min, var_max;

    var_min = var_max = extractor(cell(1));
    for (size_t i = 2; i <= NumOfCell; ++i)
    {
        const auto cur_var = extractor(cell(i));
        if (cur_var < var_min)
            var_min = cur_var;
        if (cur_var > var_max)
            var_max = cur_var;
    }
    LOG_OUT << SEP << "Min(" << var_name << ") = " << var_min << ", Max(" << var_name << ") = " << var_max << std::endl;
}

bool diagnose()
{
    stat_min_max("ConvectionFlux_X", [](const Cell &c) { return c.convection_flux.x(); });
    stat_min_max("ConvectionFlux_Y", [](const Cell &c) { return c.convection_flux.y(); });
    stat_min_max("ConvectionFlux_Z", [](const Cell &c) { return c.convection_flux.z(); });
    LOG_OUT << std::endl;
    stat_min_max("PressureFlux_X", [](const Cell &c) { return c.pressure_flux.x(); });
    stat_min_max("PressureFlux_Y", [](const Cell &c) { return c.pressure_flux.y(); });
    stat_min_max("PressureFlux_Z", [](const Cell &c) { return c.pressure_flux.z(); });
    LOG_OUT << std::endl;
    stat_min_max("ViscousFlux_X", [](const Cell &c) { return c.viscous_flux.x(); });
    stat_min_max("ViscousFlux_Y", [](const Cell &c) { return c.viscous_flux.y(); });
    stat_min_max("ViscousFlux_Z", [](const Cell &c) { return c.viscous_flux.z(); });
    LOG_OUT << std::endl;
    stat_min_max("rhoU*", [](const Cell &c) { return c.rhoU_star.x(); });
    stat_min_max("rhoV*", [](const Cell &c) { return c.rhoU_star.y(); });
    stat_min_max("rhoW*", [](const Cell &c) { return c.rhoU_star.z(); });
    LOG_OUT << std::endl;
    stat_min_max("rho", [](const Cell &c) { return c.rho; });
    stat_min_max("U", [](const Cell &c) { return c.U.x(); });
    stat_min_max("V", [](const Cell &c) { return c.U.y(); });
    stat_min_max("W", [](const Cell &c) { return c.U.z(); });
    stat_min_max("p", [](const Cell &c) { return c.p; });
    stat_min_max("T", [](const Cell &c) { return c.T; });

    return false;
}

Scalar calcTimeStep()
{
    Scalar ret = 1e-4;

    return ret;
}

void solve()
{
    int iter = 0;
    Scalar dt = 0.0; // s
    Scalar t = 0.0; // s
    bool done = false;

    LOG_OUT << std::endl << "Starting calculation ... " << std::endl;
    while (!done)
    {
        LOG_OUT << std::endl << "Iter" << ++iter << ":" << std::endl;
        dt = calcTimeStep();
        LOG_OUT << SEP << "t=" << t << "s, dt=" << dt << "s" << std::endl;
        ForwardEuler(dt);
        t += dt;
        done = diagnose();
        if (done || !(iter % OUTPUT_GAP))
        {
            updateNodalValue();
            writeTECPLOT_Nodal("flow" + std::to_string(iter) + "_NODAL.dat", "3D Cavity");
            writeTECPLOT_CellCentered("flow" + std::to_string(iter) + "_CELL.dat", "3D Cavity");
        }
        LOG_OUT << std::endl;
    }
    LOG_OUT << "Finished!" << std::endl;
}

/**
 * Initialize the computation environment.
 */
void init()
{
    static const std::string MESH_NAME = "cube32.msh";
    std::ofstream fout("Mesh Info(" + MESH_NAME + ").txt");
    LOG_OUT << std::endl << "Loading mesh \"" << MESH_NAME << "\" ... ";
    readMESH(MESH_NAME, fout);
    fout.close();
    LOG_OUT << "Done!" << std::endl;

    LOG_OUT << std::endl << "Setting B.C. of each variable ... ";
    BC_TABLE();
    LOG_OUT << "Done!" << std::endl;

    LOG_OUT << std::endl << "Preparing Least-Square coefficients used to calculate gradients ... ";
    calcLeastSquareCoef();
    LOG_OUT << "Done!" << std::endl;

    LOG_OUT << std::endl << "Preparing Pressure-Correction equation coefficients ... ";
    A_dp.resize(NumOfCell, NumOfCell);
    Q_dp.resize(NumOfCell, Eigen::NoChange);
    calcPressureCorrectionEquationCoef(A_dp);
    A_dp.makeCompressed();
    dp_solver.compute(A_dp);
    LOG_OUT << "Done!" << std::endl;

    LOG_OUT << std::endl << "Setting I.C. of each variable ... ";
    IC();
    LOG_OUT << "Done!" << std::endl;
}

/**
 * Solver entrance.
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, char *argv[])
{
    init();
    solve();

    return 0;
}

/********************************************************* END *******************************************************/
