#include <chrono>
#include <sstream>
#include <iomanip>
#include "../inc/MISC.h"

extern int NumOfPnt, NumOfFace, NumOfCell;
extern NaturalArray<Point> pnt;
extern NaturalArray<Face> face;
extern NaturalArray<Cell> cell;

Scalar duration(const clock_t &startTime, const clock_t &endTime)
{
    return static_cast<double>(endTime - startTime) / CLOCKS_PER_SEC;
}

std::string time_stamp()
{
    auto tt = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::ostringstream ss;
    ss << std::put_time(std::localtime(&tt), "%Y%m%d-%H%M%S");
    return ss.str();
}

/**
 * Calculate vectors used for NON-ORTHOGONAL correction locally.
 * @param d Local displacement vector.
 * @param S Local surface outward normal vector.
 * @param E Orthogonal part after decomposing "S".
 * @param T Non-Orthogonal part after decomposing "S", satisfying "S = E + T".
 */
void calc_noc_vec(const Vector &d, const Vector &S, Vector &E, Vector &T)
{
    E = (S.dot(S) / d.dot(S)) * d; // OverRelaxed
    T = S - E;
}

Scalar double_dot(const Tensor &A, const Tensor &B)
{
    Scalar ret = 0.0;

    for(int i = 0; i < 3; ++i)
        for(int j = 0; j < 3; ++j)
            ret += A(i, j) * B(j, i);

    return ret;
}
