#include <cmath>
#include "../inc/CHEM.h"


/**
 * Dynamic viscosity of ideal gas.
 * @param T
 * @return
 */
Scalar Sutherland(Scalar T)
{
    return 1.45e-6 * std::pow(T, 1.5) / (T + 110.0);
}
