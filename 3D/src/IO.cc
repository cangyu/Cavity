#include <iostream>
#include <iomanip>
#include <map>
#include "../inc/Miscellaneous.h"
#include "../inc/IO.h"

extern int NumOfPnt, NumOfFace, NumOfCell;
extern NaturalArray<Point> pnt;
extern NaturalArray<Face> face;
extern NaturalArray<Cell> cell;
extern NaturalArray<Patch> patch;
extern int NOC_Method;

/**
 * Load computation mesh, which is written in FLUENT format.
 * @param fin Input stream of the mesh file.
 */
void read_mesh(std::istream &fin)
{
    /// Update counting of geom elements.
    fin >> NumOfPnt >> NumOfFace >> NumOfCell;
    int NumOfPatch;
    fin >> NumOfPatch;

    /// Allocate memory for geom entities and related physical variables.
    pnt.resize(NumOfPnt);
    face.resize(NumOfFace);
    cell.resize(NumOfCell);
    patch.resize(NumOfPatch);

    /// Update nodal information.
    for (int i = 1; i <= NumOfPnt; ++i)
    {
        auto &n_dst = pnt(i);

        // 1-based global index.
        n_dst.index = i;

        // Boundary flag.
        int flag;
        fin >> flag;
        if(flag == 1)
            n_dst.at_boundary = true;
        else if(flag == 0)
            n_dst.at_boundary = false;
        else
            throw std::invalid_argument("Invalid node boundary flag.");

        // 3D location.
        fin >> n_dst.coordinate.x();
        fin >> n_dst.coordinate.y();
        fin >> n_dst.coordinate.z();

        // Adjacent nodes.
        size_t n_node;
        fin >> n_node;
        for(size_t j = 0; j < n_node; ++j)
        {
            size_t tmp;
            fin >> tmp;
        }

        // Dependent faces.
        size_t n_face;
        fin >> n_face;
        for(size_t j = 0; j < n_face; ++j)
        {
            size_t tmp;
            fin >> tmp;
        }

        // Dependent cells.
        size_t n_cell;
        fin >> n_cell;
        n_dst.dependent_cell.resize(n_cell);
        n_dst.cell_weights.resize(n_cell);
        for (size_t j = 1; j <= n_cell; ++j)
        {
            size_t tmp;
            fin >> tmp;
            n_dst.dependent_cell(j) = &cell(tmp);
        }
    }

    /// Update face information.
    for (int i = 1; i <= NumOfFace; ++i)
    {
        auto &f_dst = face(i);

        // 1-based global index.
        f_dst.index = i;

        // Boundary flag.
        int flag;
        fin >> flag;
        if(flag == 1)
            f_dst.at_boundary = true;
        else if(flag == 0)
            f_dst.at_boundary = false;
        else
            throw std::invalid_argument("Invalid face boundary flag.");

        // Shape
        int shape;
        fin >> shape;
        if(shape != 3 && shape != 4)
            throw std::invalid_argument("Unsupported face shape.");

        // Centroid
        fin >> f_dst.centroid.x();
        fin >> f_dst.centroid.y();
        fin >> f_dst.centroid.z();

        // Area.
        fin >> f_dst.area;

        // Included nodes.
        f_dst.vertex.resize(shape);
        for (int j = 1; j <= shape; ++j)
        {
            size_t tmp;
            fin >> tmp;
            f_dst.vertex(j) = &pnt(tmp);
        }

        // Adjacent cells.
        size_t c0, c1;
        fin >> c0 >> c1;
        if(c0==0)
            f_dst.c0 = nullptr;
        else
            f_dst.c0 = &cell(c0);

        if(c1==0)
            f_dst.c1 = nullptr;
        else
            f_dst.c1 = &cell(c1);

        // Unit normal vector.
        fin >> f_dst.n01.x();
        fin >> f_dst.n01.y();
        fin >> f_dst.n01.z();

        fin >> f_dst.n10.x();
        fin >> f_dst.n10.y();
        fin >> f_dst.n10.z();
    }

    /// Update cell information.
    for (int i = 1; i <= NumOfCell; ++i)
    {
        auto &c_dst = cell(i);

        // 1-based global index.
        c_dst.index = i;


        // Shape
        int shape;
        fin >> shape;
        int N1, N2;
        if(shape == 2)
        {
            N1 = 4;
            N2 = 4;
        }
        else if(shape == 4)
        {
            N1 = 8;
            N2 = 6;
        }
        else if(shape == 5)
        {
            N1 = 5;
            N2 = 5;
        }
        else if(shape == 6)
        {
            N1 = 6;
            N2 = 5;
        }
        else
            throw std::invalid_argument("Unsupported face shape.");

        // Centroid
        fin >> c_dst.centroid.x();
        fin >> c_dst.centroid.y();
        fin >> c_dst.centroid.z();

        // Volume
        fin >> c_dst.volume;

        // Included nodes.
        c_dst.vertex.resize(N1);
        for (int j = 1; j <= N1; ++j)
        {
            size_t tmp;
            fin >> tmp;
            c_dst.vertex(j) = &pnt(tmp);
        }

        // Included faces.
        c_dst.surface.resize(N2);
        for (int j = 1; j <= N2; ++j)
        {
            size_t tmp;
            fin >> tmp;
            c_dst.surface(j) = &face(tmp);
        }

        // Adjacent cells.
        c_dst.adjCell.resize(N2);
        for (int j = 1; j <= N2; ++j)
        {
            size_t tmp;
            fin >> tmp;
            c_dst.adjCell(j) = tmp==0 ? nullptr : &cell(tmp);
        }

        // Surface vectors.
        c_dst.S.resize(N2);
        for (int j = 1; j <= N2; ++j)
        {
            auto &loc_S = c_dst.S(j);
            fin >> loc_S.x();
            fin >> loc_S.y();
            fin >> loc_S.z();
            loc_S *= c_dst.surface(j)->area;
        }

        if(i == 7791)
        {
            std::cout << "\ncell " << i << std::endl;
            std::cout << c_dst.volume << std::endl;
            std::cout << "(" << c_dst.centroid.x() << ", " << c_dst.centroid.y() << ", " << c_dst.centroid.z() << ")" << std::endl;
        }

        if(i == 47721)
        {
            std::cout << "\ncell " << i << std::endl;
            std::cout << c_dst.volume << std::endl;
            std::cout << "(" << c_dst.centroid.x() << ", " << c_dst.centroid.y() << ", " << c_dst.centroid.z() << ")" << std::endl;
        }

        if(i == 47856)
        {
            std::cout << "\ncell " << i << std::endl;
            std::cout << c_dst.volume << std::endl;
            std::cout << "(" << c_dst.centroid.x() << ", " << c_dst.centroid.y() << ", " << c_dst.centroid.z() << ")" << std::endl;
        }

        if(i == 1231)
        {
            std::cout << "\ncell " << i << std::endl;
            std::cout << c_dst.volume << std::endl;
            std::cout << "(" << c_dst.centroid.x() << ", " << c_dst.centroid.y() << ", " << c_dst.centroid.z() << ")" << std::endl;
        }

        if(i == 47720)
        {
            std::cout << "\ncell " << i << std::endl;
            std::cout << c_dst.volume << std::endl;
            std::cout << "(" << c_dst.centroid.x() << ", " << c_dst.centroid.y() << ", " << c_dst.centroid.z() << ")" << std::endl;
        }
    }

    /// Nodal interpolation coefficients.
    for (int i = 1; i <= NumOfPnt; ++i)
    {
        auto &n_dst = pnt(i);
        Scalar s = 0.0;
        for (int j = 1; j <= n_dst.cell_weights.size(); ++j)
        {
            auto curAdjCell = n_dst.dependent_cell(j);
            const Scalar weighting = 1.0 / (n_dst.coordinate - curAdjCell->centroid).norm();
            n_dst.cell_weights(j) = weighting;
            s += weighting;
        }
        for (int j = 1; j <= n_dst.cell_weights.size(); ++j)
            n_dst.cell_weights(j) /= s;
    }

    /// Cell centroid to face centroid vectors and ratios.
    for (int i = 1; i <= NumOfFace; ++i)
    {
        auto &f_dst = face(i);

        /// Displacement vectors.
        if (f_dst.c0)
            f_dst.r0 = f_dst.centroid - f_dst.c0->centroid;
        else
            f_dst.r0 = ZERO_VECTOR;

        if (f_dst.c1)
            f_dst.r1 = f_dst.centroid - f_dst.c1->centroid;
        else
            f_dst.r1 = ZERO_VECTOR;

        /// Displacement ratios.
        if (f_dst.at_boundary)
        {
            if (f_dst.c0)
            {
                f_dst.ksi0 = 1.0;
                f_dst.ksi1 = 0.0;
            }
            else
            {
                f_dst.ksi0 = 0.0;
                f_dst.ksi1 = 1.0;
            }
        }
        else
        {
            const Scalar l0 = 1.0 / f_dst.r0.norm();
            const Scalar l1 = 1.0 / f_dst.r1.norm();
            const Scalar w = l0 + l1;
            f_dst.ksi0 = l0 / w;
            f_dst.ksi1 = l1 / w;
        }
    }

    /// Update boundary patch information.
    for (int i = 1; i <= NumOfPatch; ++i)
    {
        auto &p_dst = patch(i);
        fin >> p_dst.name;

        size_t n_face, n_node;
        fin >> n_face >> n_node;
        p_dst.surface.resize(n_face);
        p_dst.vertex.resize(n_node);
        for (size_t j = 0; j < n_face; ++j)
        {
            size_t tmp;
            fin >> tmp;
            auto ptr = p_dst.surface.at(j) = &face(tmp);
            ptr->parent = &p_dst;
        }
        for(size_t j = 0; j < n_node; ++j)
        {
            size_t tmp;
            fin >> tmp;
            p_dst.vertex.at(j) = &pnt(tmp);
        }
    }

    /// Vectors used for Non-Orthogonal correction within each cell.
    for (int i = 1; i <= NumOfCell; ++i)
    {
        auto &cur_cell = cell(i);
        const auto Nf = cur_cell.surface.size();
        if(cur_cell.adjCell.size() != Nf)
            throw std::runtime_error("Inconsistency detected!");

        // Allocate storage.
        cur_cell.Se.resize(Nf);
        cur_cell.St.resize(Nf);
        cur_cell.d.resize(Nf);

        // Calculate vector d, E and T.
        for(int j = 0; j < Nf; ++j)
        {
            auto cur_face = cur_cell.surface[j];
            auto cur_adj_cell = cur_cell.adjCell[j];

            // Displacement vector.
            auto &cur_d = cur_cell.d[j];
            if(cur_adj_cell == nullptr)
                cur_d = cur_face->centroid - cur_cell.centroid;
            else
                cur_d = cur_adj_cell->centroid - cur_cell.centroid;

            // Non-Orthogonal correction
            calc_noc_vec(NOC_Method, cur_d, cur_cell.S[j], cur_cell.Se[j], cur_cell.St[j]);
        }

        if(i == 47720)
        {
            std::cout << "cell " << i << std::endl;
            for(int j = 0; j < cur_cell.adjCell.size(); ++j)
            {
                std::cout << cur_cell.adjCell.at(j)->index << ": " << cur_cell.d.at(j).norm() << std::endl;
            }
        }
    }
}

void write_data(std::ostream &out, int iter, Scalar t)
{
    static const char SEP = ' ';

    out << iter << SEP << t << std::endl;
    out << NumOfPnt << SEP << NumOfFace << SEP << NumOfCell << std::endl;

    for(const auto &e : pnt)
    {
        out << e.rho << SEP;
        out << e.U.x() << SEP << e.U.y() << SEP << e.U.z() << SEP;
        out << e.p << SEP;
        out << e.T << std::endl;
    }

    for(const auto &e : face)
    {
        out << e.rho << SEP;
        out << e.U.x() << SEP << e.U.y() << SEP << e.U.z() << SEP;
        out << e.p << SEP;
        out << e.T << std::endl;
    }

    for(const auto &e : cell)
    {
        out << e.rho << SEP;
        out << e.U.x() << SEP << e.U.y() << SEP << e.U.z() << SEP;
        out << e.p << SEP;
        out << e.T << std::endl;
    }
}

void read_data(std::istream &in, int &iter, Scalar &t)
{
    in >> iter >> t;

    int n_node, n_face, n_cell;
    in >> n_node >> n_face >> n_cell;

    if(n_node != NumOfPnt || n_face != NumOfFace || n_cell != NumOfCell)
        throw std::runtime_error("Input data is not consistent with given mesh!");

    for(auto &e : pnt)
    {
        in >> e.rho;
        in >> e.U.x() >> e.U.y() >> e.U.z();
        in >> e.p;
        in >> e.T;
    }

    for(auto &e : face)
    {
        in >> e.rho;
        in >> e.U.x() >> e.U.y() >> e.U.z();
        in >> e.p;
        in >> e.T;
        e.rhoU = e.rho * e.U;
    }

    for(auto &e : cell)
    {
        in >> e.rho;
        in >> e.U.x() >> e.U.y() >> e.U.z();
        in >> e.p;
        in >> e.T;
        e.rhoU = e.rho * e.U;
    }
}
