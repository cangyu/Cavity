#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <regex>
#include <filesystem>
#include "../inc/IO.h"
#include "../inc/IC.h"
#include "../inc/BC.h"
#include "../inc/Gradient.h"
#include "../inc/PoissonEqn.h"
#include "../inc/Spatial.h"
#include "../inc/Temporal.h"
#include "../inc/Diagnose.h"
#include "../inc/MISC.h"

/***************************************************** Global Variables ***********************************************/

/* Grid utilities */
int NumOfPnt = 0, NumOfFace = 0, NumOfCell = 0;
NaturalArray<Point> pnt; /// Node objects
NaturalArray<Face> face; /// Face objects
NaturalArray<Cell> cell; /// Cell objects
NaturalArray<Patch> patch; /// Group of boundary faces

/* Flow condition */
Scalar Re = 1000.0;

/* Time-Marching */
Scalar dt = 1e-3; /// s
bool use_fixed_dt = false;

/* Global I/O style and redirection */
std::string SEP;

/* Pressure-Correction equation coefficients */
SX_MAT A_dp_2; /// The coefficient matrix
SX_VEC Q_dp_2; /// The RHS
SX_VEC x_dp_2; /// The solution
SX_AMG dp_solver_2; /// The solver object

/***************************************************** Solution Control ***********************************************/

/* I/O of mesh, case, logger and monitor */
static std::string MESH_PATH;
static std::string DATA_PATH;
static std::string RUN_TAG;
static int OUTPUT_GAP = 10;

/* Iteration timing and counting */
static size_t MAX_ITER = 100000;
static Scalar MAX_TIME = 100.0; /// s

static int iter = 0;
static Scalar t = 0.0; /// s

/******************************************************* Functions ****************************************************/

static void data_file_path(int n, std::string &fn)
{
    fn = RUN_TAG + "/ITER" + std::to_string(n) + ".txt";
}

/**
 * Directive function guiding explicit time-marching iterations.
 */
int solve()
{
    clock_t tick_begin, tick_end;
    bool done = false, diverged = false;
    Scalar single_cpu_time, total_cpu_time = 0.0;

    std::cout << "\nStarting calculation ... " << std::endl;
    while (!done)
    {
        if(iter == 0)
            prepare_first_run();

        std::cout << "\nIter" << ++iter << ":" << std::endl;
        if (!use_fixed_dt)
            dt = calcTimeStep();
        std::cout << SEP << "t=" << t << "s, dt=" << dt << "s" << std::endl;
        {
            tick_begin = clock();
            ForwardEuler(dt);
            tick_end = clock();
        }
        single_cpu_time = duration(tick_begin, tick_end);
        total_cpu_time += single_cpu_time;

        diagnose(diverged);
        if(diverged)
        {
            std::cerr << "Diverged!" << std::endl;
            return -1;
        }
        std::cout << "\n" << SEP << "CPU time: current=" << single_cpu_time << "s, total=" << total_cpu_time << "s" << std::endl;

        t += dt;
        done = iter > MAX_ITER || t > MAX_TIME;
        if (done || !(iter % OUTPUT_GAP))
        {
            INTERP_Node_Primitive();
            std::string fn;
            data_file_path(iter, fn);
            std::ofstream dts(fn);
            if(dts.fail())
                throw failed_to_open_file(fn);
            write_data(dts, iter, t);
            dts.close();
        }
    }
    std::cout << "\nFinished in " << total_cpu_time << "s!" << std::endl;

    return 0;
}

/**
 * Initialize the computation environment.
 */
void init()
{
    clock_t tick_begin, tick_end;

    std::cout << "\nLoading mesh \"" << MESH_PATH << "\" ... ";
    {
        std::ifstream ml(MESH_PATH);
        if(ml.fail())
            throw failed_to_open_file(MESH_PATH);

        tick_begin = clock();
        read_mesh(ml);
        tick_end = clock();

        ml.close();
    }
    std::cout << duration(tick_begin, tick_end) << "s" << std::endl;

    std::cout << "\nSetting B.C. type of each variable ... ";
    BC_TABLE();
    std::cout << "Done!" << std::endl;

    std::cout << "\nPreparing coefficients ... ";
    {
        tick_begin = clock();
        prepare_lsq();
        prepare_gg();
        prepare_gpc_rm();
        tick_end = clock();
    }
    std::cout << duration(tick_begin, tick_end) << "s" << std::endl;

    std::cout << "\nPreparing Pressure-Correction equation coefficients ... ";
    {
        Q_dp_2 = sx_vec_create(NumOfCell);
        x_dp_2 = sx_vec_create(NumOfCell);

        tick_begin = clock();
        calcPressureCorrectionEquationCoef(A_dp_2);
        tick_end = clock();
    }
    std::cout << duration(tick_begin, tick_end) << "s" << std::endl;

    prepare_dp_solver(A_dp_2, dp_solver_2);

    if(DATA_PATH.empty())
    {
        std::cout << "\nSetting I.C. ... ";
        tick_begin = clock();
        IC();
        tick_end = clock();
        std::cout << duration(tick_begin, tick_end) << "s" << std::endl;

        std::cout << "\nWriting initial output ... ";
        std::string fn;
        data_file_path(0, fn);
        std::ofstream dts(fn);
        if(dts.fail())
            throw failed_to_open_file(fn);
        write_data(dts, 0, 0.0);
        dts.close();
    }
    else
    {
        std::cout << "\nSetting I.C. from \"" + DATA_PATH + "\" ... ";
        std::ifstream dts(DATA_PATH);
        if(dts.fail())
            throw failed_to_open_file(DATA_PATH);
        tick_begin = clock();
        read_data(dts, iter, t);
        tick_end = clock();
        std::cout << duration(tick_begin, tick_end) << "s" << std::endl;
        dts.close();
    }
    std::cout << "Done!" << std::endl;
}

/**
 * Solver entrance.
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, char *argv[])
{
    SEP = "  ";
    bool need_to_create_folder = true;
    bool resume_mode = false;

    /* Parse parameters */
    int cnt = 1;
    while (cnt < argc)
    {
        if (!std::strcmp(argv[cnt], "--mesh"))
            MESH_PATH = argv[cnt + 1];
        else if (!std::strcmp(argv[cnt], "--data"))
            DATA_PATH = argv[cnt + 1]; /// Using I.C. from certain data file.
        else if (!std::strcmp(argv[cnt], "--tag"))
            RUN_TAG = argv[cnt + 1];
        else if (!std::strcmp(argv[cnt], "--iteration"))
            MAX_ITER = std::strtoul(argv[cnt + 1], nullptr, 10);
        else if (!std::strcmp(argv[cnt], "--time-span"))
            MAX_TIME = std::strtod(argv[cnt + 1], nullptr); /// In seconds by default.
        else if (!std::strcmp(argv[cnt], "--write-interval"))
            OUTPUT_GAP = (int)std::strtol(argv[cnt + 1], nullptr, 10);
        else if (!std::strcmp(argv[cnt], "--Re"))
            Re = std::strtod(argv[cnt + 1], nullptr);
        else if(!std::strcmp(argv[cnt], "--resume-from"))
        {
            need_to_create_folder = false;
            resume_mode = true;
            RUN_TAG = argv[cnt + 1];
        }
        else if (!std::strcmp(argv[cnt], "--time-step"))
        {
            use_fixed_dt = true;
            dt = std::strtod(argv[cnt + 1], nullptr); /// In seconds by default.
        }
        else
            throw std::invalid_argument("Unrecognized option: \"" + std::string(argv[cnt]) + "\".");

        cnt += 2;
    }

    /* Establish output destination */
    if (RUN_TAG.empty())
        RUN_TAG = time_stamp();
    
    if(need_to_create_folder)
    {
        auto ret = std::filesystem::create_directory(RUN_TAG);
        if (!ret)
        {
            std::cerr << "Failed to create output directory!" << std::endl;
            return -1;
        }
    }

    /* Continuation */
    if(resume_mode)
    {
        // TODO
    }

    /* Report */
    std::cout << "Output directory set to: \"" << RUN_TAG << "\"" << std::endl;
    std::cout << "\nRe=" << Re << std::endl;
    if (use_fixed_dt)
        std::cout << "\nUsing fixed time-step: " << dt << std::endl;
    std::cout << "\nMax iterations: " << MAX_ITER << std::endl;
    std::cout << "\nMax run time: " << MAX_TIME << "s" << std::endl;
    std::cout << "\nRecord solution every " << OUTPUT_GAP << " iteration" << std::endl;

    /* Initialize environment */
    init();

    /* Transient iteration loop */
    solve();

    /* Finalize */
    sx_vec_destroy(&x_dp_2);
    sx_vec_destroy(&Q_dp_2);
    sx_mat_destroy(&A_dp_2);
    sx_amg_data_destroy(&dp_solver_2);

    return 0;
}

/********************************************************* END ********************************************************/
