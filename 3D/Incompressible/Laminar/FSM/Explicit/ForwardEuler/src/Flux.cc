#include "../inc/custom_type.h"
#include "../inc/Flux.h"

extern int NumOfPnt, NumOfFace, NumOfCell;
extern NaturalArray<Point> pnt;
extern NaturalArray<Face> face;
extern NaturalArray<Cell> cell;
extern NaturalArray<Patch> patch;

static void calcCellContinuityFlux()
{
    // TODO
}

static void calcCellMomentumFlux()
{
    for (auto &c : cell)
    {
        c.pressure_flux.setZero();
        c.convection_flux.setZero();
        c.viscous_flux.setZero();

        for (size_t j = 0; j < c.S.size(); ++j)
        {
            auto f = c.surface.at(j);
            const auto &S_f = c.S.at(j);

            // convection term
            const Vector cur_convection_flux = f->rhoU * f->U.dot(S_f);
            c.convection_flux += cur_convection_flux;

            // pressure term
            const Vector cur_pressure_flux = f->p * S_f;
            c.pressure_flux += cur_pressure_flux;

            // viscous term
            const Vector cur_viscous_flux = { S_f.dot(f->tau.col(0)), S_f.dot(f->tau.col(1)), S_f.dot(f->tau.col(2)) };
            c.viscous_flux += cur_viscous_flux;
        }
    }
}

static void calcCellEnergyFlux()
{
    // TODO
}

void calcCellFlux()
{
    // Continuity equation
    calcCellContinuityFlux();

    // Momentum equation
    calcCellMomentumFlux();

    // Energy equation
    calcCellEnergyFlux();
}
