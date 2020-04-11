#include <iostream>
#include <fstream>
#include <map>
#include <cmath>
#include <functional>
#include <filesystem>
#include <cstdio>
#include "../inc/IO.h"
#include "../inc/IC.h"
#include "../inc/BC.h"
#include "../inc/LeastSquare.h"
#include "../inc/PoissonEqn.h"
#include "../inc/Discretization.h"
#include "../inc/Miscellaneous.h"

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

/* I/O of mesh, case, logger and monitor */
static const std::string MESH_DIR = "mesh/";
static const std::string MESH_NAME = "cube32.msh";
static const std::string RUN_TAG = time_stamp_str();
static const std::string NODAL_CASE_DIR = RUN_TAG + "/Nodal/";
static const std::string CENTERED_CASE_DIR = RUN_TAG + "/Centered/";
static const size_t OUTPUT_GAP = 5;
static std::ostream &LOG_OUT = std::cout;
static const std::string SEP = "  ";
static clock_t tick_begin, tick_end;

/* Iteration timing and counting */
static const int MAX_ITER = 2000;
static const Scalar MAX_TIME = 100.0; // s

/***************************************************** Solution Control **********************************************/

Scalar calcTimeStep()
{
    Scalar ret = 1e-2;

    return ret;
}

static void write_flowfield(int n, double t)
{
    static char tmp[5];
    std::snprintf(tmp, sizeof(tmp) / sizeof(char), "%04d", n);
    const std::string CASE_NAME = "Iter" + std::string(tmp);
    static const std::string CASE_SUFFIX = ".dat";

    const std::string NODAL_CASE_PATH = NODAL_CASE_DIR + CASE_NAME + CASE_SUFFIX;
    const std::string CENTERED_CASE_PATH = CENTERED_CASE_DIR + CASE_NAME + CASE_SUFFIX;

    static const std::string CASE_TITLE = "3D Incompressible Unsteady Laminar Cavity Flow";
    static const std::string ZONE_TEXT = "Entire Domain";

    updateNodalValue();
    writeTECPLOT_Nodal(NODAL_CASE_PATH, CASE_TITLE, ZONE_TEXT, t);
    writeTECPLOT_Centered(CENTERED_CASE_PATH, CASE_TITLE, ZONE_TEXT, t);
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

    if (var_name == "p")
    {
        LOG_OUT.setf(std::ios::fixed);
        auto w = LOG_OUT.precision(10);
        LOG_OUT << SEP << "Min(" << var_name << ") = " << var_min << ", Max(" << var_name << ") = " << var_max << std::endl;
        LOG_OUT.precision(w);
        LOG_OUT.unsetf(std::ios::fixed);
    }
    else
        LOG_OUT << SEP << "Min(" << var_name << ") = " << var_min << ", Max(" << var_name << ") = " << var_max << std::endl;
}

static double stat_div(const Cell &c)
{
    double ret = 0.0;
    const auto Nf = c.surface.size();
    for (int i = 0; i < Nf; ++i)
    {
        auto curFace = c.surface.at(i);
        ret += curFace->rhoU.dot(c.S.at(i));
    }
    ret /= c.volume;
    return ret;
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
    stat_min_max("rhoU*_X", [](const Cell &c) { return c.rhoU_star.x(); });
    stat_min_max("rhoU*_Y", [](const Cell &c) { return c.rhoU_star.y(); });
    stat_min_max("rhoU*_Z", [](const Cell &c) { return c.rhoU_star.z(); });
    LOG_OUT << std::endl;
    stat_min_max("rho", [](const Cell &c) { return c.rho; });
    stat_min_max("U_X", [](const Cell &c) { return c.U.x(); });
    stat_min_max("U_Y", [](const Cell &c) { return c.U.y(); });
    stat_min_max("U_Z", [](const Cell &c) { return c.U.z(); });
    stat_min_max("p", [](const Cell &c) { return c.p; });
    stat_min_max("p'", [](const Cell &c) { return c.p_prime; });
    stat_min_max("T", [](const Cell &c) { return c.T; });
    LOG_OUT << std::endl;
    stat_min_max("div", stat_div);
    stat_min_max("CFL", [](const Cell &c) { return c.U.norm() * calcTimeStep() * 32; });

    return false;
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
        tick_begin = clock();
        ForwardEuler(dt);
        tick_end = clock();
        t += dt;
        done = diagnose();
        LOG_OUT << std::endl << SEP << duration(tick_begin, tick_end) << "s used." << std::endl;
        if (done || !(iter % OUTPUT_GAP))
            write_flowfield(iter, t);
    }
    LOG_OUT << "Finished!" << std::endl;
}

/**
 * Initialize the computation environment.
 */
void init()
{
    std::filesystem::create_directory(RUN_TAG);

    const std::string fn_mesh_log = RUN_TAG + "/MeshDesc.txt";
    std::ofstream fout(fn_mesh_log);
    if (fout.fail())
        throw std::runtime_error("Failed to open target log file for mesh.");
    LOG_OUT << std::endl << "Loading mesh \"" << MESH_NAME << "\" ... ";
    tick_begin = clock();
    readMESH(MESH_DIR + MESH_NAME, fout);
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

    std::filesystem::create_directory(NODAL_CASE_DIR);
    std::filesystem::create_directory(CENTERED_CASE_DIR);

    LOG_OUT << std::endl << "Writting initial output ... ";
    write_flowfield(0, 0.0);
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
