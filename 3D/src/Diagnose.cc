#include <iostream>
#include <cmath>
#include <functional>
#include "../inc/Diagnose.h"

extern int NumOfPnt, NumOfFace, NumOfCell;
extern NaturalArray<Point> pnt;
extern NaturalArray<Face> face;
extern NaturalArray<Cell> cell;
extern NaturalArray<Patch> patch;
extern Scalar dt;
extern Scalar Re;

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

    if (var_name == "p" || var_name == "p'")
    {
        auto w = std::cout.precision(10);
        std::cout << "Min(" << var_name << ") = " << var_min << ", Max(" << var_name << ") = " << var_max << std::endl;
        std::cout.precision(w);
    }
    else
        std::cout << "Min(" << var_name << ") = " << var_min << ", Max(" << var_name << ") = " << var_max << std::endl;
}

static Scalar max_div;
static size_t max_div_idx;

static Scalar stat_div(const Cell &c)
{
    Scalar ret = 0.0;
    const auto Nf = c.surface.size();
    for (int i = 0; i < Nf; ++i)
    {
        auto curFace = c.surface.at(i);
        ret += curFace->rhoU.dot(c.S.at(i));
    }
    ret /= c.volume;

    if(std::abs(ret) > max_div)
    {
        max_div = ret;
        max_div_idx = c.index;
    }

    return ret;
}

static Scalar stat_cfl(const Cell &c)
{
    static const Scalar A = 3 * std::sqrt(3);

    return A * c.grad_U.norm() * dt;
}

void diagnose(bool &diverge_flag)
{
    std::cout << std::endl;
    stat_min_max("rho", [](const Cell &c) { return c.rho; });
    stat_min_max("U_X", [](const Cell &c) { return c.U.x(); });
    stat_min_max("U_Y", [](const Cell &c) { return c.U.y(); });
    stat_min_max("U_Z", [](const Cell &c) { return c.U.z(); });
    stat_min_max("p", [](const Cell &c) { return c.p; });
    stat_min_max("p'", [](const Cell &c) { return c.p_prime; });
    stat_min_max("T", [](const Cell &c) { return c.T; });
    std::cout << std::endl;

    stat_min_max("CFL", stat_cfl);

    max_div = 0.0;
    stat_min_max("div", stat_div);
    diverge_flag = std::fabs(max_div) > 1e2;
}

/**
 * Transient time-step for each explicit marching iteration.
 * @return Current time-step used for temporal integration.
 */
Scalar calcTimeStep()
{
    static const Scalar CFL = 0.5;
    Scalar momentum_max_dt = std::numeric_limits<Scalar>::max();
    Scalar energy_max_dt = std::numeric_limits<Scalar>::max();

    for(const auto &c : cell)
    {
        const Scalar L = std::pow(c.volume, 1.0 / 3);
        const Scalar L2 = L * L;

        /// Momentum
        Scalar momentum_dt = 1.0 / (3.0 * c.U.norm() / L + 2.0 / Re * 3 / L2);
        if(momentum_dt < momentum_max_dt)
            momentum_max_dt = momentum_dt;

        /// Energy
        Scalar energy_dt = 1.0 / (3.0 * c.U.norm() / L + 2.0 * c.conductivity / (c.rho * c.specific_heat_p) * 3 / (L * L));
        if(energy_dt < energy_max_dt)
            energy_max_dt = energy_dt;
    }
    momentum_max_dt *= CFL;
    energy_max_dt *= CFL;
    std::cout << "\nMomentum max time-step:" << momentum_max_dt;
    std::cout << "\nEnergy max time-step:" << energy_max_dt;
    std::cout << std::endl;
    return std::min(momentum_max_dt, energy_max_dt);
}
