#include "../inc/custom_type.h"
#include "../inc/LeastSquare.h"

extern size_t NumOfPnt, NumOfFace, NumOfCell;
extern NaturalArray<Point> pnt;
extern NaturalArray<Face> face;
extern NaturalArray<Cell> cell;
extern NaturalArray<Patch> patch;

/**
 * Convert Eigen's intrinsic QR decomposition matrix into R^-1 * Q^T
 * @param J
 * @param J_INV
 */
static void extractQRMatrix(const Eigen::Matrix<Scalar, Eigen::Dynamic, 3> &J, Eigen::Matrix<Scalar, 3, Eigen::Dynamic> &J_INV)
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

    Mat J_rho;
    std::array<Mat, 3> J_U;
    Mat J_p, J_p_prime;
    Mat J_T;

    for (auto &c : cell)
    {
        const size_t nF = c.surface.size();

        J_rho.resize(nF, Eigen::NoChange);
        J_U[0].resize(nF, Eigen::NoChange);
        J_U[1].resize(nF, Eigen::NoChange);
        J_U[2].resize(nF, Eigen::NoChange);
        J_p.resize(nF, Eigen::NoChange);
        J_p_prime.resize(nF, Eigen::NoChange);
        J_T.resize(nF, Eigen::NoChange);

        for (size_t j = 0; j < nF; ++j)
        {
            auto curFace = c.surface.at(j);
            if (curFace->atBdry)
            {
                const auto dx = curFace->center.x() - c.center.x();
                const auto dy = curFace->center.y() - c.center.y();
                const auto dz = curFace->center.z() - c.center.z();

                const auto dx2 = 2 * dx;
                const auto dy2 = 2 * dy;
                const auto dz2 = 2 * dz;

                // Density
                switch (curFace->rho_BC)
                {
                case Dirichlet:
                    J_rho.row(j) << dx, dy, dz;
                    break;
                case Neumann:
                    J_rho.row(j) << dx2, dy2, dz2;
                    break;
                case Robin:
                    throw unsupported_boundary_condition(Robin);
                default:
                    break;
                }

                // Velocity-X
                switch (curFace->U_BC[0])
                {
                case Dirichlet:
                    J_U[0].row(j) << dx, dy, dz;
                    break;
                case Neumann:
                    J_U[0].row(j) << dx2, dy2, dz2;
                    break;
                case Robin:
                    throw unsupported_boundary_condition(Robin);
                default:
                    break;
                }

                // Velocity-Y
                switch (curFace->U_BC[1])
                {
                case Dirichlet:
                    J_U[1].row(j) << dx, dy, dz;
                    break;
                case Neumann:
                    J_U[1].row(j) << dx2, dy2, dz2;
                    break;
                case Robin:
                    throw unsupported_boundary_condition(Robin);
                default:
                    break;
                }

                // Velocity-Z
                switch (curFace->U_BC[2])
                {
                case Dirichlet:
                    J_U[2].row(j) << dx, dy, dz;
                    break;
                case Neumann:
                    J_U[2].row(j) << dx2, dy2, dz2;
                    break;
                case Robin:
                    throw unsupported_boundary_condition(Robin);
                default:
                    break;
                }

                // Pressure
                switch (curFace->p_BC)
                {
                case Dirichlet:
                    J_p.row(j) << dx, dy, dz;
                    break;
                case Neumann:
                    J_p.row(j) << dx2, dy2, dz2;
                    break;
                case Robin:
                    throw unsupported_boundary_condition(Robin);
                default:
                    break;
                }

                // Pressure-Correction
                switch (curFace->p_prime_BC)
                {
                case Dirichlet:
                    J_p_prime.row(j) << dx, dy, dz;
                    break;
                case Neumann:
                    J_p_prime.row(j) << dx2, dy2, dz2;
                    break;
                case Robin:
                    throw unsupported_boundary_condition(Robin);
                default:
                    break;
                }

                // Temperature
                switch (curFace->T_BC)
                {
                case Dirichlet:
                    J_T.row(j) << dx, dy, dz;
                    break;
                case Neumann:
                    J_T.row(j) << dx2, dy2, dz2;
                    break;
                case Robin:
                    throw unsupported_boundary_condition(Robin);
                default:
                    break;
                }
            }
            else
            {
                auto curAdjCell = c.adjCell.at(j);

                const auto dx = curAdjCell->center.x() - c.center.x();
                const auto dy = curAdjCell->center.y() - c.center.y();
                const auto dz = curAdjCell->center.z() - c.center.z();

                // Density
                J_rho.row(j) << dx, dy, dz;

                // Velocity-X
                J_U[0].row(j) << dx, dy, dz;

                // Velocity-Y
                J_U[1].row(j) << dx, dy, dz;

                // Velocity-Z
                J_U[2].row(j) << dx, dy, dz;

                // Pressure
                J_p.row(j) << dx, dy, dz;

                // Pressure-Correction
                J_p_prime.row(j) << dx, dy, dz;

                // Temperature
                J_T.row(j) << dx, dy, dz;
            }
        }

        // Density
        extractQRMatrix(J_rho, c.J_INV_rho);

        // Velocity-X
        extractQRMatrix(J_U[0], c.J_INV_U[0]);

        // Velocity-Y
        extractQRMatrix(J_U[1], c.J_INV_U[1]);

        // Velocity-Z
        extractQRMatrix(J_U[2], c.J_INV_U[2]);

        // Pressure
        extractQRMatrix(J_p, c.J_INV_p);

        // Pressure-Correction
        extractQRMatrix(J_p_prime, c.J_INV_p_prime);

        // Temperature
        extractQRMatrix(J_T, c.J_INV_T);
    }
}
