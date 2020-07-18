#include "../inc/custom_type.h"
#include "../inc/Flux.h"

extern int NumOfPnt;
extern int NumOfFace;
extern int NumOfCell;
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

        const auto Nf = c.S.size();
        for (int j = 0; j < Nf; ++j)
        {
            auto f = c.surface.at(j);
            const auto &Sf = c.S.at(j);

            /// convection term
            const Vector cur_convection_flux = f->rhoU * f->U.dot(Sf);
            c.convection_flux += cur_convection_flux;

            /// pressure term
            const Vector cur_pressure_flux = f->p * Sf;
            c.pressure_flux += cur_pressure_flux;

            /// viscous term
            const Vector cur_viscous_flux = f->tau * Sf;
            c.viscous_flux += cur_viscous_flux;
        }
    }
}

static void calcCellEnergyFlux()
{
    // TODO
}

void calc_cell_flux()
{
    /// Continuity equation
    calcCellContinuityFlux();

    /// Momentum equation
    calcCellMomentumFlux();

    /// Energy equation
    calcCellEnergyFlux();
}
