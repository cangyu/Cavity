#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include "../inc/IO.h"
#include "../inc/IC.h"
#include "../inc/BC.h"
#include "../inc/LeastSquare.h"
#include "../inc/PoissonEqn.h"
#include "../inc/Discretization.h"
#include "../inc/Diagnose.h"
#include "../inc/Miscellaneous.h"

/***************************************************** Global Variables ***********************************************/

/* Grid utilities */
int NumOfPnt = 0, NumOfFace = 0, NumOfCell = 0;
NaturalArray<Point> pnt; /// Node objects
NaturalArray<Face> face; /// Face objects
NaturalArray<Cell> cell; /// Cell objects
NaturalArray<Patch> patch; /// Group of boundary faces

/* Global I/O style and redirection */
std::string SEP;
std::ostream &LOG_OUT = std::cout;

/* Pressure-Correction equation coefficients */
SX_MAT A_dp_2; /// The coefficient matrix
SX_VEC Q_dp_2; /// The RHS
SX_AMG dp_solver_2; /// The solver object

/***************************************************** Solution Control ***********************************************/

/* I/O of mesh, case, logger and monitor */
static std::string MESH_PATH;
static std::string DATA_PATH;
static std::string RUN_TAG;
static int OUTPUT_GAP = 10;

/* Iteration timing and counting */
static int MAX_ITER = 100000;
static Scalar MAX_TIME = 100.0; /// s

/******************************************************* Functions ****************************************************/

/**
 * Transient time-step for each explicit marching iteration.
 * @return Current time-step used for temporal integration.
 */
Scalar calcTimeStep()
{
    Scalar ret = 1e-3;
    return ret;
}

/**
 * Directive function guiding explicit time-marching iterations.
 */
void solve()
{
    clock_t tick_begin, tick_end;
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
        done = iter > MAX_ITER || t > MAX_TIME;
        diagnose();
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
    /// Separation style
    SEP = "  ";

    /// Timing vars
    clock_t tick_begin, tick_end;

    /// Check tag
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
        else if(!std::strcmp(argv[cnt], "--time"))
            MAX_TIME = std::atof(argv[cnt+1]); /// In seconds by default.
        else if(!std::strcmp(argv[cnt], "--interval"))
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
