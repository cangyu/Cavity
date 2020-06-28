#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <functional>
#include <filesystem>
#include "../inc/IO.h"
#include "../inc/IC.h"
#include "../inc/BC.h"
#include "../inc/LeastSquare.h"
#include "../inc/PoissonEqn.h"
#include "../inc/Discretization.h"
#include "../inc/Miscellaneous.h"

/* Grid utilities */
int NumOfPnt = 0, NumOfFace = 0, NumOfCell = 0;
NaturalArray<Point> pnt; /// Node objects
NaturalArray<Face> face; /// Face objects
NaturalArray<Cell> cell; /// Cell objects
NaturalArray<Patch> patch; /// Group of boundary faces

/* Pressure-Correction equation coefficients */
SX_MAT A_dp_2; /// The coefficient matrix
SX_VEC Q_dp_2; /// The RHS
SX_AMG dp_solver_2; /// The solver object

/* I/O of mesh, case, logger and monitor */
std::string MESH_PATH;
std::string DATA_PATH;
std::string RUN_TAG;
int OUTPUT_GAP = 20;
static std::ostream &LOG_OUT = std::cout;
static const std::string SEP = "  ";
static clock_t tick_begin, tick_end;

/* Iteration timing and counting */
int MAX_ITER = 20000;
Scalar MAX_TIME = 100.0; /// s

/***************************************************** Solution Control ***********************************************/

Scalar calcTimeStep()
{
    Scalar ret = 2.5e-3;

    return ret;
}

static void stat_min_max(const std::string& var_name, const std::function<Scalar(const Cell&)> &extractor)
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

bool diagnose(int n, Scalar t)
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
    stat_min_max("CFL", [](const Cell &c) { return c.U.norm() * calcTimeStep() * 64; });

    return n > MAX_ITER || t > MAX_TIME;
}

void solve()
{
    int iter = 0;
    Scalar dt; /// s
    Scalar t = 0.0; /// s
    bool done = false;

    LOG_OUT << "\nStarting calculation ... " << std::endl;
    while (!done)
    {
        LOG_OUT << "\nIter" << ++iter << ":" << std::endl;
        dt = calcTimeStep();
        LOG_OUT << SEP << "t=" << t << "s, dt=" << dt << "s" << std::endl;
        tick_begin = clock();
        ForwardEuler(dt);
        tick_end = clock();
        t += dt;
        done = diagnose(iter, t);
        LOG_OUT << "\n" << SEP << duration(tick_begin, tick_end) << "s used." << std::endl;
        if (done || !(iter % OUTPUT_GAP))
            record_computation_domain(RUN_TAG, iter, t);
    }
    LOG_OUT << "\nFinished!" << std::endl;
}

/**
 * Initialize the computation environment.
 */
void init()
{
    if(RUN_TAG.empty())
        RUN_TAG = time_stamp_str();

    if(std::filesystem::create_directory(RUN_TAG))
        LOG_OUT << "Output directory set to: \"" << RUN_TAG << "\"" << std::endl;
    else
        throw std::runtime_error("Failed to create output directory.");

    LOG_OUT << "\nLoading mesh \"" << MESH_PATH << "\" ... ";
    const std::string fn_mesh_log = RUN_TAG + "/MeshDesc.txt";
    std::ofstream ml_out(fn_mesh_log);
    if (ml_out.fail())
        throw failed_to_open_file(fn_mesh_log);
    tick_begin = clock();
    read_fluent_mesh(MESH_PATH, ml_out);
    tick_end = clock();
    ml_out.close();
    LOG_OUT << duration(tick_begin, tick_end) << "s" << std::endl;

    LOG_OUT << "\nSetting B.C. of each variable ... ";
    BC_TABLE();
    LOG_OUT << "Done!" << std::endl;

    LOG_OUT << "\nPreparing Least-Square coefficients ... ";
    tick_begin = clock();
    calcLeastSquareCoef();
    tick_end = clock();
    LOG_OUT << duration(tick_begin, tick_end) << "s" << std::endl;

    LOG_OUT << "\nPreparing Pressure-Correction equation coefficients ... ";
    Q_dp_2 = sx_vec_create(NumOfCell);
    tick_begin = clock();
    calcPressureCorrectionEquationCoef(A_dp_2);
    tick_end = clock();
    LOG_OUT << duration(tick_begin, tick_end) << "s" << std::endl;

    prepare_dp_solver(A_dp_2, dp_solver_2);

    LOG_OUT << "\nSetting I.C. of each variable ... ";
    IC();
    LOG_OUT << "Done!" << std::endl;

    LOG_OUT << "\nWriting initial output ... ";
    record_computation_domain(RUN_TAG, 0, 0.0);
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
    /* Parse parameters */
    int cnt = 1;
    while(cnt < argc)
    {
        if(!std::strcmp(argv[cnt], "--mesh") || !std::strcmp(argv[cnt], "-m"))
            MESH_PATH = argv[cnt+1];
        else if(!std::strcmp(argv[cnt], "--data"))
            DATA_PATH = argv[cnt+1]; /// Using I.C. from certain data file.
        else if(!std::strcmp(argv[cnt], "--tag"))
            RUN_TAG = argv[cnt+1];
        else if(!std::strcmp(argv[cnt], "--iter"))
            MAX_ITER = std::atoi(argv[cnt+1]);
        else if(!std::strcmp(argv[cnt], "--duration"))
            MAX_TIME = std::atof(argv[cnt+1]); /// In seconds by default.
        else if(!std::strcmp(argv[cnt], "--record_interval"))
            OUTPUT_GAP = std::atoi(argv[cnt+1]);
        else
        {
            const std::string opt = argv[cnt];
            throw std::invalid_argument("Unrecognized option: \"" + opt + "\".");
        }
        cnt += 2;
    }
    
    /* Initialize environment */
    init();

    /* Transient iteration loop */
    solve();

    /* Finalize */
    return 0;
}

/********************************************************* END ********************************************************/
