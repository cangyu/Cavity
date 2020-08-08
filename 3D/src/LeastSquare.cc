#include "../inc/custom_type.h"
#include "../inc/LeastSquare.h"

extern int NumOfPnt;
extern int NumOfFace;
extern int NumOfCell;
extern NaturalArray<Point> pnt;
extern NaturalArray<Face> face;
extern NaturalArray<Cell> cell;
extern NaturalArray<Patch> patch;

/**
 * Convert Eigen's intrinsic QR decomposition matrix into R^-1 * Q^T
 * @param J The coefficient matrix to be factorized.
 * @param J_INV The general inverse of input matrix using QR decomposition.
 */
static void extract_qr_matrix
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
 */
void calc_least_square_coefficient_matrix()
{
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 3> Mat;

    Mat J_rho;
    std::array<Mat, 3> J_U;
    Mat J_p, J_p_prime;
    Mat J_T;

    for (auto &c : cell)
    {
        const auto nF = c.surface.size();

        J_rho.resize(nF, Eigen::NoChange);
        J_U[0].resize(nF, Eigen::NoChange);
        J_U[1].resize(nF, Eigen::NoChange);
        J_U[2].resize(nF, Eigen::NoChange);
        J_p.resize(nF, Eigen::NoChange);
        J_p_prime.resize(nF, Eigen::NoChange);
        J_T.resize(nF, Eigen::NoChange);

        for (int j = 0; j < nF; ++j)
        {
            /// Possible coefficients for current face
            auto curFace = c.surface.at(j);
            Vector d, n;
            if (curFace->at_boundary)
            {
                d = curFace->centroid - c.centroid;
                n = c.S.at(j) / curFace->area;

                /// Density
                switch (curFace->parent->rho_BC)
                {
                case Dirichlet:
                    J_rho.row(j) << d.x(), d.y(), d.z();
                    break;
                case Neumann:
                    J_rho.row(j) << n.x(), n.y(), n.z();
                    break;
                case Robin:
                    throw robin_bc_is_not_supported();
                default:
                    break;
                }

                /// Velocity-X
                switch (curFace->parent->U_BC[0])
                {
                case Dirichlet:
                    J_U[0].row(j) << d.x(), d.y(), d.z();
                    break;
                case Neumann:
                    J_U[0].row(j) << n.x(), n.y(), n.z();
                    break;
                case Robin:
                    throw robin_bc_is_not_supported();
                default:
                    break;
                }

                /// Velocity-Y
                switch (curFace->parent->U_BC[1])
                {
                case Dirichlet:
                    J_U[1].row(j) << d.x(), d.y(), d.z();
                    break;
                case Neumann:
                    J_U[1].row(j) << n.x(), n.y(), n.z();
                    break;
                case Robin:
                    throw robin_bc_is_not_supported();
                default:
                    break;
                }

                /// Velocity-Z
                switch (curFace->parent->U_BC[2])
                {
                case Dirichlet:
                    J_U[2].row(j) << d.x(), d.y(), d.z();
                    break;
                case Neumann:
                    J_U[2].row(j) << n.x(), n.y(), n.z();
                    break;
                case Robin:
                    throw robin_bc_is_not_supported();
                default:
                    break;
                }

                /// Pressure
                switch (curFace->parent->p_BC)
                {
                case Dirichlet:
                    J_p.row(j) << d.x(), d.y(), d.z();
                    break;
                case Neumann:
                    J_p.row(j) << n.x(), n.y(), n.z();
                    break;
                case Robin:
                    throw robin_bc_is_not_supported();
                default:
                    break;
                }

                /// Pressure-Correction
                switch (curFace->parent->p_prime_BC)
                {
                case Dirichlet:
                    J_p_prime.row(j) << d.x(), d.y(), d.z();
                    break;
                case Neumann:
                    J_p_prime.row(j) << n.x(), n.y(), n.z();
                    break;
                case Robin:
                    throw robin_bc_is_not_supported();
                default:
                    break;
                }

                /// Temperature
                switch (curFace->parent->T_BC)
                {
                case Dirichlet:
                    J_T.row(j) << d.x(), d.y(), d.z();
                    break;
                case Neumann:
                    J_T.row(j) << n.x(), n.y(), n.z();
                    break;
                case Robin:
                    throw robin_bc_is_not_supported();
                default:
                    break;
                }
            }
            else
            {
                auto curAdjCell = c.adjCell.at(j);
                d = curAdjCell->centroid - c.centroid;

                /// Density
                J_rho.row(j) << d.x(), d.y(), d.z();

                /// Velocity-X
                J_U[0].row(j) << d.x(), d.y(), d.z();

                /// Velocity-Y
                J_U[1].row(j) << d.x(), d.y(), d.z();

                /// Velocity-Z
                J_U[2].row(j) << d.x(), d.y(), d.z();

                /// Pressure
                J_p.row(j) << d.x(), d.y(), d.z();

                /// Pressure-Correction
                J_p_prime.row(j) << d.x(), d.y(), d.z();

                /// Temperature
                J_T.row(j) << d.x(), d.y(), d.z();
            }
        }

        /// Density
        extract_qr_matrix(J_rho, c.J_INV_rho);

        /// Velocity-X
        extract_qr_matrix(J_U[0], c.J_INV_U[0]);

        /// Velocity-Y
        extract_qr_matrix(J_U[1], c.J_INV_U[1]);

        /// Velocity-Z
        extract_qr_matrix(J_U[2], c.J_INV_U[2]);

        /// Pressure
        extract_qr_matrix(J_p, c.J_INV_p);

        /// Pressure-Correction
        extract_qr_matrix(J_p_prime, c.J_INV_p_prime);

        /// Temperature
        extract_qr_matrix(J_T, c.J_INV_T);
    }
}
