#include <iostream>
#include <fstream>
#include <map>
#include <cmath>
#include <functional>
#include <ctime>
#include "../inc/IO.h"
#include "../inc/IC.h"
#include "../inc/BC.h"
#include "../inc/LeastSquare.h"
#include "../inc/PoissonEqn.h"
#include "../inc/Discretization.h"

/* Grid utilities */
size_t NumOfPnt = 0;
size_t NumOfFace = 0;
size_t NumOfCell = 0;

NaturalArray<Point> pnt; // Node objects
NaturalArray<Face> face; // Face objects
NaturalArray<Cell> cell; // Cell objects
NaturalArray<Patch> patch; // Group of boundary faces

/* Pressure-Corrrection equation coefficients */
Eigen::SparseMatrix<Scalar> A_dp;
Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Q_dp;
Eigen::BiCGSTAB<Eigen::SparseMatrix<Scalar>, Eigen::IncompleteLUT<Scalar>> dp_solver;

/* I/O of logger and monitor */
static std::ostream &LOG_OUT = std::cout;
static const std::string SEP = "  ";
static clock_t tick_begin, tick_end;

/***************************************************** Solution Control **********************************************/

static inline double duration(const clock_t &startTime, const clock_t &endTime)
{
    return (endTime - startTime) * 1.0 / CLOCKS_PER_SEC;
}

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
    LOG_OUT << std::endl;
    stat_min_max("div", [](const Cell &c) { return c.grad_U.trace(); });
    stat_min_max("CFL", [](const Cell &c) { return c.U.norm() * 5e-3 * 64; });

    return false;
}

Scalar calcTimeStep()
{
    Scalar ret = 5e-3;

    return ret;
}

void solve()
{
    static const size_t OUTPUT_GAP = 5;

    /* Iteration timing and counting */
    const int MAX_ITER = 2000;
    const Scalar MAX_TIME = 100.0; // s

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
        tick_begin = clock();
        ForwardEuler(dt);
        tick_end = clock();
        t += dt;
        done = diagnose();
        LOG_OUT << std::endl << SEP << duration(tick_begin, tick_end) << "s used." << std::endl;
        if (done || !(iter % OUTPUT_GAP))
        {
            updateNodalValue();
            writeTECPLOT_Nodal("flow" + std::to_string(iter) + "_NODAL.dat", "3D Cavity");
            writeTECPLOT_CellCentered("flow" + std::to_string(iter) + "_CELL.dat", "3D Cavity");
        }
    }
    LOG_OUT << "Finished!" << std::endl;
}

/**
 * Initialize the computation environment.
 */
void init()
{
    LOG_OUT << Eigen::nbThreads() << " threads used." << std::endl;

    static const std::string MESH_NAME = "cube32.msh";
    std::ofstream fout("Mesh Info(" + MESH_NAME + ").txt");
    if (fout.fail())
        throw std::runtime_error("Failed to open target log file for mesh.");
    LOG_OUT << std::endl << "Loading mesh \"" << MESH_NAME << "\" ... ";
    tick_begin = clock();
    readMESH(MESH_NAME, fout);
    tick_end = clock();
    fout.close();
    LOG_OUT << duration(tick_begin, tick_end) << "s" << std::endl;

    LOG_OUT << std::endl << "Setting B.C. of each variable ... ";
    BC_TABLE();
    LOG_OUT << "Done!" << std::endl;

    LOG_OUT << std::endl << "Preparing Least-Square coefficients ... ";
    tick_begin = clock();
    calcLeastSquareCoef();
    tick_end = clock();
    LOG_OUT << duration(tick_begin, tick_end) << "s" << std::endl;

    LOG_OUT << std::endl << "Preparing Pressure-Correction equation coefficients ... ";
    A_dp.resize(NumOfCell, NumOfCell);
    Q_dp.resize(NumOfCell, Eigen::NoChange);
    tick_begin = clock();
    calcPressureCorrectionEquationCoef(A_dp);
    A_dp.makeCompressed();
    tick_end = clock();
    LOG_OUT << duration(tick_begin, tick_end) << "s" << std::endl;

    LOG_OUT << std::endl << "Matrix factorization ... ";
    tick_begin = clock();
    dp_solver.compute(A_dp);
    tick_end = clock();
    LOG_OUT << duration(tick_begin, tick_end) << "s" << std::endl;

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
