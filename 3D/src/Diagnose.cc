#include <functional>
#include "../inc/custom_type.h"
#include "../inc/Diagnose.h"

extern int NumOfPnt, NumOfFace, NumOfCell;
extern NaturalArray<Point> pnt;
extern NaturalArray<Face> face;
extern NaturalArray<Cell> cell;
extern NaturalArray<Patch> patch;
extern std::string SEP;
extern std::ostream &LOG_OUT;
extern Scalar dt;

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
        auto w = LOG_OUT.precision(10);
        LOG_OUT << SEP << "Min(" << var_name << ") = " << var_min << ", Max(" << var_name << ") = " << var_max << std::endl;
        LOG_OUT.precision(w);
    }
    else
        LOG_OUT << SEP << "Min(" << var_name << ") = " << var_min << ", Max(" << var_name << ") = " << var_max << std::endl;
}

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
    return ret;
}

static Scalar stat_cfl(const Cell &c)
{
    return c.grad_U.norm() * dt;
}

void diagnose()
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
    stat_min_max("CFL", stat_cfl);
}
