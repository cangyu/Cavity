#include <iomanip>
#include <fstream>
#include <map>
#include "../inc/IO.h"

extern int NumOfPnt, NumOfFace, NumOfCell;
extern NaturalArray<Point> pnt;
extern NaturalArray<Face> face;
extern NaturalArray<Cell> cell;
extern NaturalArray<Patch> patch;
extern int NOC_Method;

enum class TECPLOT_FE_MESH_TYPE : int
{
    TET = 1,
    HEX = 2,
    POLY = 3
};

/// Type of mesh
static TECPLOT_FE_MESH_TYPE IO_MT;

/// Composition of mesh
static int n_tet = -1;
static int n_hex = -1;
static int n_pyramid = -1;
static int n_prism = -1;

/**
 * Calculate vectors used for NON-ORTHOGONAL correction locally.
 * @param opt Choice of method.
 * 1 - Minimum Correction
 * 2 - Orthogonal Correction
 * 3 - Over-Relaxed Correction
 * @param d Local displacement vector.
 * @param S Local surface outward normal vector.
 * @param E Orthogonal part after decomposing "S".
 * @param T Non-Orthogonal part after decomposing "S", satisfying "S = E + T".
 */
static void calc_noc_vector(int opt, const Vector &d, const Vector &S, Vector &E, Vector &T)
{
    Vector e = d;
    e /= d.norm();
    const Scalar S_mod = S.norm();
    const Scalar cos_theta = e.dot(S) / S_mod;

    if(opt == 1)
        E = S_mod * cos_theta * e;
    else if(opt == 2)
        E = S_mod * e;
    else if(opt == 3)
        E = S_mod / cos_theta * e;
    else
        throw std::invalid_argument("Invalid NON-ORTHOGONAL correction option!");

    T = S - E;
}

/**
 * Load computation mesh, which is written in FLUENT format.
 * @param MESH_PATH Path to target mesh file.
 * @param LOG_OUT Output stream used to hold progress indications.
 */
void read_mesh(const std::string &MESH_PATH, std::ostream &LOG_OUT)
{
    std::ifstream fin(MESH_PATH);
    if(fin.fail())
        throw std::runtime_error("Failed to open input mesh file.");

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
    n_tet = n_hex = n_prism = n_pyramid = 0;
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
            ++n_tet;
            N1 = 4;
            N2 = 4;
        }
        else if(shape == 4)
        {
            ++n_hex;
            N1 = 8;
            N2 = 6;
        }
        else if(shape == 5)
        {
            ++n_pyramid;
            N1 = 5;
            N2 = 5;
        }
        else if(shape == 6)
        {
            ++n_prism;
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
    }

    /// Update mesh type flag.
    if(n_tet == NumOfCell)
        IO_MT = TECPLOT_FE_MESH_TYPE::TET;
    else if(n_hex == NumOfCell)
        IO_MT = TECPLOT_FE_MESH_TYPE::HEX;
    else
        IO_MT = TECPLOT_FE_MESH_TYPE::POLY;

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
            calc_noc_vector(NOC_Method, cur_d, cur_cell.S[j], cur_cell.Se[j], cur_cell.St[j]);
        }
    }
}

static void formatted_block_data_writer
(
    std::ostream &out,
    const std::vector<Scalar> &val,
    size_t nRec1Line,
    const std::string &sep,
    bool nlAtLast
)
{
    size_t i = 0;
    for (const auto &e : val)
    {
        out << sep << e;
        ++i;
        if (i % nRec1Line == 0)
            out << std::endl;
    }
    if (nlAtLast && (val.size() % nRec1Line))
        out << std::endl;
}

static void write_tec_grid_tet(const std::string &fn, const std::string &title)
{
    /// Format param
    static const size_t RECORD_PER_LINE = 10;
    static const std::string SEP = " ";

    /// Open target file.
    std::ofstream fout(fn);
    if (fout.fail())
        throw failed_to_open_file(fn);

    /// File Header
    fout << "TITLE=\"" << title << "\"" << std::endl;
    fout << "FILETYPE=GRID" << std::endl;
    fout << R"(VARIABLES="X", "Y", "Z")" << std::endl;

    /// Zone Record
    /// Volume Zone
    fout << "ZONE T=\"" << "INTERIOR" << "\"" << std::endl;
    fout << "STRANDID=" << 1 << std::endl;
    fout << "NODES=" << NumOfPnt << std::endl;
    fout << "ELEMENTS=" << NumOfCell << std::endl;
    fout << "ZONETYPE=" << "FETETRAHEDRON" << std::endl;
    fout << "DATAPACKING=BLOCK" << std::endl;
    fout << "VARLOCATION=([1-3]=NODAL)" << std::endl;

    /// X-Coordinates
    for (size_t i = 1; i <= NumOfPnt; ++i)
    {
        fout << SEP << pnt(i).coordinate.x();
        if (i % RECORD_PER_LINE == 0)
            fout << std::endl;
    }
    if (NumOfPnt % RECORD_PER_LINE != 0)
        fout << std::endl;

    /// Y-Coordinates
    for (size_t i = 1; i <= NumOfPnt; ++i)
    {
        fout << SEP << pnt(i).coordinate.y();
        if (i % RECORD_PER_LINE == 0)
            fout << std::endl;
    }
    if (NumOfPnt % RECORD_PER_LINE != 0)
        fout << std::endl;

    /// Z-Coordinates
    for (size_t i = 1; i <= NumOfPnt; ++i)
    {
        fout << SEP << pnt(i).coordinate.z();
        if (i % RECORD_PER_LINE == 0)
            fout << std::endl;
    }
    if (NumOfPnt % RECORD_PER_LINE != 0)
        fout << std::endl;

    /// Connectivity
    for (size_t i = 1; i <= NumOfCell; ++i)
    {
        const auto &v = cell(i).vertex;
        if(v.size() != 4)
            throw insufficient_vertexes(i);

        for (const auto &e : v)
            fout << SEP << e->index;
        fout << std::endl;
    }

    /// Surface Zone
    for(int k = 1; k <= patch.size(); ++k)
    {
        const auto &p = patch(k);
        const auto &s = p.surface;
        const auto &v = p.vertex;

        /// Record local order of nodes.
        std::map<int, size_t> n2n;
        size_t cnt = 0;
        for(auto f : s)
        {
            const auto &vl = f->vertex;
            if(vl.size() != 3)
                throw insufficient_vertexes(f->index);

            for(auto e : vl)
            {
                const int ci = e->index;
                auto it = n2n.find(ci);
                if(it == n2n.end())
                {
                    ++cnt;
                    n2n[ci] = cnt;
                }
            }
        }

        /// Check
        if(v.size() != n2n.size())
            throw std::runtime_error("Inconsistent with previous calculation.");

        /// Zone Header
        fout << "ZONE T=\"" << p.name << "\"" << std::endl;
        fout << "STRANDID=" << 1 + k << std::endl;
        fout << "NODES=" << v.size() << std::endl;
        fout << "ELEMENTS=" << s.size() << std::endl;
        fout << "ZONETYPE=" << "FETRIANGLE" << std::endl;
        fout << "DATAPACKING=BLOCK" << std::endl;
        fout << "VARLOCATION=([1-3]=NODAL)" << std::endl;

        /// Gather coordinate components.
        std::vector<Scalar> cord_comp_x(v.size(), 0.0);
        std::vector<Scalar> cord_comp_y(v.size(), 0.0);
        std::vector<Scalar> cord_comp_z(v.size(), 0.0);
        for(size_t i = 0; i < v.size(); ++i)
        {
            auto pt = v[i];
            cord_comp_x[i] = pt->coordinate.x();
            cord_comp_y[i] = pt->coordinate.y();
            cord_comp_z[i] = pt->coordinate.z();

        }

        /// X-Coordinates
        formatted_block_data_writer(fout, cord_comp_x, RECORD_PER_LINE, SEP, true);

        /// Y-Coordinates
        formatted_block_data_writer(fout, cord_comp_y, RECORD_PER_LINE, SEP, true);

        /// Z-Coordinates
        formatted_block_data_writer(fout, cord_comp_z, RECORD_PER_LINE, SEP, true);

        /// Connectivity
        for (auto f : s)
        {
            const auto &vl = f->vertex;
            for (auto e : vl)
                fout << SEP << n2n[e->index];
            fout << std::endl;
        }
    }

    /// Finalize
    fout.close();
}

static void write_tec_grid_hex(const std::string &fn, const std::string &title)
{
    /// Format param
    static const size_t RECORD_PER_LINE = 10;
    static const std::string SEP = " ";

    /// Open target file.
    std::ofstream fout(fn);
    if (fout.fail())
        throw failed_to_open_file(fn);

    /// File Header
    fout << "TITLE=\"" << title << "\"" << std::endl;
    fout << "FILETYPE=GRID" << std::endl;
    fout << R"(VARIABLES="X", "Y", "Z")" << std::endl;

    /// Zone Record
    /// Volume Zone
    fout << "ZONE T=\"" << "INTERIOR" << "\"" << std::endl;
    fout << "STRANDID=" << 1 << std::endl;
    fout << "NODES=" << NumOfPnt << std::endl;
    fout << "ELEMENTS=" << NumOfCell << std::endl;
    fout << "ZONETYPE=" << "FEBRICK" << std::endl;
    fout << "DATAPACKING=BLOCK" << std::endl;
    fout << "VARLOCATION=([1-3]=NODAL)" << std::endl;

    /// X-Coordinates
    for (size_t i = 1; i <= NumOfPnt; ++i)
    {
        fout << SEP << pnt(i).coordinate.x();
        if (i % RECORD_PER_LINE == 0)
            fout << std::endl;
    }
    if (NumOfPnt % RECORD_PER_LINE != 0)
        fout << std::endl;

    /// Y-Coordinates
    for (size_t i = 1; i <= NumOfPnt; ++i)
    {
        fout << SEP << pnt(i).coordinate.y();
        if (i % RECORD_PER_LINE == 0)
            fout << std::endl;
    }
    if (NumOfPnt % RECORD_PER_LINE != 0)
        fout << std::endl;

    /// Z-Coordinates
    for (size_t i = 1; i <= NumOfPnt; ++i)
    {
        fout << SEP << pnt(i).coordinate.z();
        if (i % RECORD_PER_LINE == 0)
            fout << std::endl;
    }
    if (NumOfPnt % RECORD_PER_LINE != 0)
        fout << std::endl;

    /// Connectivity
    for (size_t i = 1; i <= NumOfCell; ++i)
    {
        const auto &v = cell(i).vertex;
        if(v.size() != 8)
            throw insufficient_vertexes(i);

        for (const auto &e : v)
            fout << SEP << e->index;
        fout << std::endl;
    }

    /// Surface Zone
    for(int k = 1; k <= patch.size(); ++k)
    {
        const auto &p = patch(k);
        const auto &s = p.surface;
        const auto &v = p.vertex;

        /// Record local order of nodes.
        std::map<int, size_t> n2n;
        size_t cnt = 0;
        for(auto f : s)
        {
            const auto &vl = f->vertex;
            if(vl.size() != 4)
                throw insufficient_vertexes(f->index);

            for(auto e : vl)
            {
                const int ci = e->index;
                auto it = n2n.find(ci);
                if(it == n2n.end())
                {
                    ++cnt;
                    n2n[ci] = cnt;
                }
            }
        }

        /// Check
        if(v.size() != n2n.size())
            throw std::runtime_error("Inconsistent with previous calculation.");

        /// Zone Header
        fout << "ZONE T=\"" << p.name << "\"" << std::endl;
        fout << "STRANDID=" << 1 + k << std::endl;
        fout << "NODES=" << v.size() << std::endl;
        fout << "ELEMENTS=" << s.size() << std::endl;
        fout << "ZONETYPE=" << "FEQUADRILATERAL" << std::endl;
        fout << "DATAPACKING=BLOCK" << std::endl;
        fout << "VARLOCATION=([1-3]=NODAL)" << std::endl;

        /// Gather coordinate components.
        std::vector<Scalar> cord_comp_x(v.size(), 0.0);
        std::vector<Scalar> cord_comp_y(v.size(), 0.0);
        std::vector<Scalar> cord_comp_z(v.size(), 0.0);
        for(size_t i = 0; i < v.size(); ++i)
        {
            auto pt = v[i];
            cord_comp_x[i] = pt->coordinate.x();
            cord_comp_y[i] = pt->coordinate.y();
            cord_comp_z[i] = pt->coordinate.z();
        }

        /// X-Coordinates
        formatted_block_data_writer(fout, cord_comp_x, RECORD_PER_LINE, SEP, true);

        /// Y-Coordinates
        formatted_block_data_writer(fout, cord_comp_y, RECORD_PER_LINE, SEP, true);

        /// Z-Coordinates
        formatted_block_data_writer(fout, cord_comp_z, RECORD_PER_LINE, SEP, true);

        /// Connectivity
        for (auto f : s)
        {
            const auto &vl = f->vertex;
            for (auto e : vl)
                fout << SEP << n2n[e->index];
            fout << std::endl;
        }
    }

    /// Finalize
    fout.close();
}

static void write_tec_grid_poly(const std::string &fn, const std::string &title)
{
    throw std::runtime_error("Polyhedral grid is NOT supported currently!");
}

/**
 * Write out the computation grid in TECPLOT format.
 * For data sharing when performing transient simulation.
 * @param fn
 * @param type
 * @param title
 */
void write_tec_grid(const std::string &fn, const std::string &title)
{
    switch(IO_MT)
    {
    case TECPLOT_FE_MESH_TYPE::TET:
        write_tec_grid_tet(fn, title);
        break;
    case TECPLOT_FE_MESH_TYPE::HEX:
        write_tec_grid_hex(fn, title);
        break;
    case TECPLOT_FE_MESH_TYPE::POLY:
        write_tec_grid_poly(fn, title);
        break;
    default:
        throw std::invalid_argument("Invalid specification of mesh type!");
    }
}

static void write_tec_solution_tet(const std::string &fn, double t, const std::string &title)
{
    /// Format param
    static const size_t RECORD_PER_LINE = 10;
    static const std::string SEP = " ";

    /// Open target file.
    std::ofstream fout(fn);
    if (fout.fail())
        throw failed_to_open_file(fn);

    /// File Header
    fout << "TITLE=\"" << title << "\"" << std::endl;
    fout << "FILETYPE=SOLUTION" << std::endl;
    fout << R"(VARIABLES="rho", "U", "V", "W", "P", "T")" << std::endl;

    /// Zone Record
    /// Volume Zone
    fout << "ZONE T=\"" << "INTERIOR" << "\"" << std::endl;
    fout << "STRANDID=" << 1 << std::endl;
    fout << "SOLUTIONTIME=" << t << std::endl;
    fout << "NODES=" << NumOfPnt << std::endl;
    fout << "ELEMENTS=" << NumOfCell << std::endl;
    fout << "ZONETYPE=" << "FETETRAHEDRON" << std::endl;
    fout << "DATAPACKING=BLOCK" << std::endl;
    fout << "VARLOCATION=([1-6]=CELLCENTERED)" << std::endl;

    /// Density
    for (size_t i = 1; i <= NumOfCell; ++i)
    {
        fout << SEP << cell(i).rho0;
        if (i % RECORD_PER_LINE == 0)
            fout << std::endl;
    }
    if (NumOfCell % RECORD_PER_LINE != 0)
        fout << std::endl;

    /// Velocity-X
    for (size_t i = 1; i <= NumOfCell; ++i)
    {
        fout << SEP << cell(i).U0.x();
        if (i % RECORD_PER_LINE == 0)
            fout << std::endl;
    }
    if (NumOfCell % RECORD_PER_LINE != 0)
        fout << std::endl;

    /// Velocity-Y
    for (size_t i = 1; i <= NumOfCell; ++i)
    {
        fout << SEP << cell(i).U0.y();
        if (i % RECORD_PER_LINE == 0)
            fout << std::endl;
    }
    if (NumOfCell % RECORD_PER_LINE != 0)
        fout << std::endl;

    /// Velocity-Z
    for (size_t i = 1; i <= NumOfCell; ++i)
    {
        fout << SEP << cell(i).U0.z();
        if (i % RECORD_PER_LINE == 0)
            fout << std::endl;
    }
    if (NumOfCell % RECORD_PER_LINE != 0)
        fout << std::endl;

    /// Pressure
    fout.setf(std::ios::fixed);
    for (size_t i = 1; i <= NumOfCell; ++i)
    {
        fout << SEP << std::setprecision(10) << cell(i).p0;
        if (i % RECORD_PER_LINE == 0)
            fout << std::endl;
    }
    fout.unsetf(std::ios::fixed);
    if (NumOfCell % RECORD_PER_LINE != 0)
        fout << std::endl;

    /// Temperature
    for (size_t i = 1; i <= NumOfCell; ++i)
    {
        fout << SEP << cell(i).T0;
        if (i % RECORD_PER_LINE == 0)
            fout << std::endl;
    }
    if (NumOfCell % RECORD_PER_LINE != 0)
        fout << std::endl;

    /// Surface Zone
    for(int k = 1; k <= patch.size(); ++k)
    {
        const auto &p = patch(k);
        const auto &s = p.surface;
        const auto &v = p.vertex;

        /// Zone Header
        fout << "ZONE T=\"" << p.name << "\"" << std::endl;
        fout << "STRANDID=" << 1 + k << std::endl;
        fout << "SOLUTIONTIME=" << t << std::endl;
        fout << "NODES=" << v.size() << std::endl;
        fout << "ELEMENTS=" << s.size() << std::endl;
        fout << "ZONETYPE=" << "FETRIANGLE" << std::endl;
        fout << "DATAPACKING=BLOCK" << std::endl;
        fout << "VARLOCATION=([1-6]=CELLCENTERED)" << std::endl;

        /// Density
        int pos = 0;
        for (auto f : s)
        {
            ++pos;
            fout << SEP << f->rho;
            if (pos % RECORD_PER_LINE == 0)
                fout << std::endl;
        }
        if (pos % RECORD_PER_LINE)
            fout << std::endl;

        /// Velocity-X
        pos = 0;
        for (auto f : s)
        {
            ++pos;
            fout << SEP << f->U.x();
            if (pos % RECORD_PER_LINE == 0)
                fout << std::endl;
        }
        if (pos % RECORD_PER_LINE)
            fout << std::endl;

        /// Velocity-Y
        pos = 0;
        for (auto f : s)
        {
            ++pos;
            fout << SEP << f->U.y();
            if (pos % RECORD_PER_LINE == 0)
                fout << std::endl;
        }
        if (pos % RECORD_PER_LINE)
            fout << std::endl;

        /// Velocity-Z
        pos = 0;
        for (auto f : s)
        {
            ++pos;
            fout << SEP << f->U.z();
            if (pos % RECORD_PER_LINE == 0)
                fout << std::endl;
        }
        if (pos % RECORD_PER_LINE)
            fout << std::endl;

        /// Pressure
        pos = 0;
        fout.setf(std::ios::fixed);
        for (auto f : s)
        {
            ++pos;
            fout << SEP << std::setprecision(10) << f->p;
            if (pos % RECORD_PER_LINE == 0)
                fout << std::endl;
        }
        fout.unsetf(std::ios::fixed);
        if (pos % RECORD_PER_LINE)
            fout << std::endl;

        /// Temperature
        pos = 0;
        for (auto f : s)
        {
            ++pos;
            fout << SEP << f->T;
            if (pos % RECORD_PER_LINE == 0)
                fout << std::endl;
        }
        if (pos % RECORD_PER_LINE)
            fout << std::endl;
    }

    /// Finalize
    fout.close();
}

static void write_tec_solution_hex(const std::string &fn, double t, const std::string &title)
{
    /// Format param
    static const size_t RECORD_PER_LINE = 10;
    static const std::string SEP = " ";

    /// Open target file.
    std::ofstream fout(fn);
    if (fout.fail())
        throw failed_to_open_file(fn);

    /// File Header
    fout << "TITLE=\"" << title << "\"" << std::endl;
    fout << "FILETYPE=SOLUTION" << std::endl;
    fout << R"(VARIABLES="rho", "U", "V", "W", "P", "T")" << std::endl;

    /// Zone Record
    /// Volume Zone
    fout << "ZONE T=\"" << "INTERIOR" << "\"" << std::endl;
    fout << "STRANDID=" << 1 << std::endl;
    fout << "SOLUTIONTIME=" << t << std::endl;
    fout << "NODES=" << NumOfPnt << std::endl;
    fout << "ELEMENTS=" << NumOfCell << std::endl;
    fout << "ZONETYPE=" << "FEBRICK" << std::endl;
    fout << "DATAPACKING=BLOCK" << std::endl;
    fout << "VARLOCATION=([1-6]=CELLCENTERED)" << std::endl;

    /// Density
    for (size_t i = 1; i <= NumOfCell; ++i)
    {
        fout << SEP << cell(i).rho0;
        if (i % RECORD_PER_LINE == 0)
            fout << std::endl;
    }
    if (NumOfCell % RECORD_PER_LINE != 0)
        fout << std::endl;

    /// Velocity-X
    for (size_t i = 1; i <= NumOfCell; ++i)
    {
        fout << SEP << cell(i).U0.x();
        if (i % RECORD_PER_LINE == 0)
            fout << std::endl;
    }
    if (NumOfCell % RECORD_PER_LINE != 0)
        fout << std::endl;

    /// Velocity-Y
    for (size_t i = 1; i <= NumOfCell; ++i)
    {
        fout << SEP << cell(i).U0.y();
        if (i % RECORD_PER_LINE == 0)
            fout << std::endl;
    }
    if (NumOfCell % RECORD_PER_LINE != 0)
        fout << std::endl;

    /// Velocity-Z
    for (size_t i = 1; i <= NumOfCell; ++i)
    {
        fout << SEP << cell(i).U0.z();
        if (i % RECORD_PER_LINE == 0)
            fout << std::endl;
    }
    if (NumOfCell % RECORD_PER_LINE != 0)
        fout << std::endl;

    /// Pressure
    fout.setf(std::ios::fixed);
    for (size_t i = 1; i <= NumOfCell; ++i)
    {
        fout << SEP << std::setprecision(10) << cell(i).p0;
        if (i % RECORD_PER_LINE == 0)
            fout << std::endl;
    }
    fout.unsetf(std::ios::fixed);
    if (NumOfCell % RECORD_PER_LINE != 0)
        fout << std::endl;

    /// Temperature
    for (size_t i = 1; i <= NumOfCell; ++i)
    {
        fout << SEP << cell(i).T0;
        if (i % RECORD_PER_LINE == 0)
            fout << std::endl;
    }
    if (NumOfCell % RECORD_PER_LINE != 0)
        fout << std::endl;

    /// Surface Zone
    for(int k = 1; k <= patch.size(); ++k)
    {
        const auto &p = patch(k);
        const auto &s = p.surface;
        const auto &v = p.vertex;

        /// Zone Header
        fout << "ZONE T=\"" << p.name << "\"" << std::endl;
        fout << "STRANDID=" << 1 + k << std::endl;
        fout << "SOLUTIONTIME=" << t << std::endl;
        fout << "NODES=" << v.size() << std::endl;
        fout << "ELEMENTS=" << s.size() << std::endl;
        fout << "ZONETYPE=" << "FEQUADRILATERAL" << std::endl;
        fout << "DATAPACKING=BLOCK" << std::endl;
        fout << "VARLOCATION=([1-6]=CELLCENTERED)" << std::endl;

        /// Density
        int pos = 0;
        for (auto f : s)
        {
            ++pos;
            fout << SEP << f->rho;
            if (pos % RECORD_PER_LINE == 0)
                fout << std::endl;
        }
        if (pos % RECORD_PER_LINE)
            fout << std::endl;

        /// Velocity-X
        pos = 0;
        for (auto f : s)
        {
            ++pos;
            fout << SEP << f->U.x();
            if (pos % RECORD_PER_LINE == 0)
                fout << std::endl;
        }
        if (pos % RECORD_PER_LINE)
            fout << std::endl;

        /// Velocity-Y
        pos = 0;
        for (auto f : s)
        {
            ++pos;
            fout << SEP << f->U.y();
            if (pos % RECORD_PER_LINE == 0)
                fout << std::endl;
        }
        if (pos % RECORD_PER_LINE)
            fout << std::endl;

        /// Velocity-Z
        pos = 0;
        for (auto f : s)
        {
            ++pos;
            fout << SEP << f->U.z();
            if (pos % RECORD_PER_LINE == 0)
                fout << std::endl;
        }
        if (pos % RECORD_PER_LINE)
            fout << std::endl;

        /// Pressure
        pos = 0;
        fout.setf(std::ios::fixed);
        for (auto f : s)
        {
            ++pos;
            fout << SEP << std::setprecision(10) << f->p;
            if (pos % RECORD_PER_LINE == 0)
                fout << std::endl;
        }
        fout.unsetf(std::ios::fixed);
        if (pos % RECORD_PER_LINE)
            fout << std::endl;

        /// Temperature
        pos = 0;
        for (auto f : s)
        {
            ++pos;
            fout << SEP << f->T;
            if (pos % RECORD_PER_LINE == 0)
                fout << std::endl;
        }
        if (pos % RECORD_PER_LINE)
            fout << std::endl;
    }

    /// Finalize
    fout.close();
}

static void write_tec_solution_poly(const std::string &fn, double t, const std::string &title)
{
    throw std::runtime_error("Polyhedral solution is NOT supported currently!");
}

/**
 * Output computation results.
 * Cell-Centered values are exported, including boundary variables.
 * Only for continuation purpose.
 * @param fn
 * @param type
 * @param t
 * @param title
 */
void write_tec_solution(const std::string &fn, double t, const std::string &title)
{
    switch(IO_MT)
    {
    case TECPLOT_FE_MESH_TYPE::TET:
        write_tec_solution_tet(fn, t, title);
        break;
    case TECPLOT_FE_MESH_TYPE::HEX:
        write_tec_solution_hex(fn, t, title);
        break;
    case TECPLOT_FE_MESH_TYPE::POLY:
        write_tec_solution_poly(fn, t, title);
        break;
    default:
        throw std::invalid_argument("Invalid specification of mesh type!");
    }
}

/**
 * High-level function used for the solution procedure to record both grid and data automatically.
 * @param prefix Folder where records of current run are stored.
 * @param n Iteration counter.
 * @param t Solution time, or the time elapsed relative to I.C. prescribed by a data file.
 */
void record_computation_domain(const std::string &prefix, int n, Scalar t)
{
    static const std::string GRID_TITLE = "GRID";
    const std::string SOLUTION_TITLE = "ITER" + std::to_string(n);

    static const std::string GRID_PATH = prefix + "/GRID.dat";
    const std::string SOLUTION_PATH = prefix + "/ITER" + std::to_string(n) + ".dat";

    if(n == 0)
        write_tec_grid(GRID_PATH, GRID_TITLE);

    write_tec_solution(SOLUTION_PATH, t, SOLUTION_TITLE);
}

/**
 * Extract CELL-CENTERED solution variables only.
 * Should be consistent with existing mesh!!!
 * Only for continuation purpose.
 * @param fn
 */
void read_tec_solution(const std::string &fn)
{
    /// Open target file
    std::ifstream fin(fn);
    if (fin.fail())
        throw failed_to_open_file(fn);

    /// Skip header
    for (int i = 0; i < 11; ++i)
    {
        std::string tmp;
        std::getline(fin, tmp);
    }

    /// Load data.
    /// ONLY take internal fields!!!

    /// Density
    for (size_t i = 1; i <= NumOfCell; ++i)
        fin >> cell(i).rho0;

    /// Velocity-X
    for (size_t i = 1; i <= NumOfCell; ++i)
        fin >> cell(i).U0.x();

    /// Velocity-Y
    for (size_t i = 1; i <= NumOfCell; ++i)
        fin >> cell(i).U0.y();

    /// Velocity-Z
    for (size_t i = 1; i <= NumOfCell; ++i)
        fin >> cell(i).U0.z();

    /// Pressure
    for (size_t i = 1; i <= NumOfCell; ++i)
        fin >> cell(i).p0;

    /// Temperature
    for (size_t i = 1; i <= NumOfCell; ++i)
        fin >> cell(i).T0;

    /// Finalize
    fin.close();
}
