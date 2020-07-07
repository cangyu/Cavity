#include "../inc/custom_type.h"
#include "../inc/LeastSquare.h"

extern int NumOfPnt, NumOfFace, NumOfCell;
extern NaturalArray<Point> pnt;
extern NaturalArray<Face> face;
extern NaturalArray<Cell> cell;
extern NaturalArray<Patch> patch;

/**
 * Convert Eigen's intrinsic QR decomposition matrix into R^-1 * Q^T
 * @param J The coefficient matrix to be factorized.
 * @param J_INV The general inverse of input matrix using QR decomposition.
 */
static void extractQRMatrix
(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 3> &J, 
    Eigen::Matrix<Scalar, 3, Eigen::Dynamic> &J_INV
)
{
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Mat;

    auto QR = J.householderQr();
    const Mat Q = QR.householderQ();
    const Mat R = QR.matrixQR().triangularView<Eigen::Upper>();

    const Mat Q0 = Q.block(0, 0, J.rows(), 3);
    const Mat R0 = R.block<3, 3>(0, 0);

    J_INV = R0.inverse() * Q0.transpose();
}

/**
 * QR decomposition matrix of each cell.
 * Ghost cells are used when the B.C. of boundary faces are set to Neumann type.
 */
void calcLeastSquareCoef()
{
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 3> Mat;

    Mat J;

    for (auto &c : cell)
    {
        const auto nF = c.surface.size();

        J.resize(nF, Eigen::NoChange);

        for (int j = 0; j < nF; ++j)
        {
            auto curFace = c.surface.at(j);

            Scalar dx, dy, dz;
            if (curFace->atBdry)
            {
                dx = curFace->center.x() - c.center.x();
                dy = curFace->center.y() - c.center.y();
                dz = curFace->center.z() - c.center.z();
            }
            else
            {
                auto curAdjCell = c.adjCell.at(j);

                dx = curAdjCell->center.x() - c.center.x();
                dy = curAdjCell->center.y() - c.center.y();
                dz = curAdjCell->center.z() - c.center.z();
            }

            J.row(j) << dx, dy, dz;
        }

        extractQRMatrix(J, c.J_INV);
    }
}
