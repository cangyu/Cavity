#include <map>
#include "../3rd_party/TYDF/inc/xf.h"
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
static void calc_non_orthogonal_correction_vector
(
    int opt,
    const Vector &d,
    const Vector &S,
    Vector &E,
    Vector &T
)
{
    const Vector e = d / d.norm();
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
void read_fluent_mesh(const std::string &MESH_PATH, std::ostream &LOG_OUT)
{
    using namespace GridTool;

    /// An external package is used to load Fluent mesh.
    const XF::MESH mesh(MESH_PATH, LOG_OUT);

    /// Update counting of geom elements.
    NumOfPnt = mesh.numOfNode();
    NumOfFace = mesh.numOfFace();
    NumOfCell = mesh.numOfCell();

    /// Update composition
    n_tet = n_hex = n_prism = n_pyramid = 0;
    for(int i = 1; i <= NumOfCell; ++i)
    {
        const auto &e = mesh.cell(i);
        switch(e.type)
        {
        case XF::CELL::HEXAHEDRAL:
            ++n_hex;
            break;
        case XF::CELL::TETRAHEDRAL:
            ++n_tet;
            break;
        case XF::CELL::PYRAMID:
            ++n_pyramid;
            break;
        case XF::CELL::WEDGE:
            ++n_prism;
            break;
        default:
            throw std::runtime_error("Unexpected cell element type.");
        }
    }

    /// Update mesh type flag
    if(n_tet == NumOfCell)
        IO_MT = TECPLOT_FE_MESH_TYPE::TET;
    else if(n_hex == NumOfCell)
        IO_MT = TECPLOT_FE_MESH_TYPE::HEX;
    else
        IO_MT = TECPLOT_FE_MESH_TYPE::POLY;

    /// Allocate memory for geom entities and related physical variables.
    pnt.resize(NumOfPnt);
    face.resize(NumOfFace);
    cell.resize(NumOfCell);

    /// Update nodal information.
    for (int i = 1; i <= NumOfPnt; ++i)
    {
        const auto &n_src = mesh.node(i);
        auto &n_dst = pnt(i);

        // 1-based global index.
        n_dst.index = i;

        // 3D location.
        const auto &c_src = n_src.coordinate;
        n_dst.coordinate = { c_src.x(), c_src.y(), c_src.z() };

        // Boundary flag.
        n_dst.atBdry = n_src.atBdry;

        // Adjacent cells.
        // Used for interpolation from cell-centered to nodal.
        n_dst.dependentCell.resize(n_src.dependentCell.size());
        n_dst.cellWeightingCoef.resize(n_src.dependentCell.size());
        for (auto j = 1; j <= n_src.dependentCell.size(); ++j)
            n_dst.dependentCell(j) = &cell(n_src.dependentCell(j));
    }

    /// Update face information.
    for (int i = 1; i <= NumOfFace; ++i)
    {
        const auto &f_src = mesh.face(i);
        auto &f_dst = face(i);

        // Assign face index.
        f_dst.index = i;
        f_dst.atBdry = f_src.atBdry;

        // Face center location.
        const auto &c_src = f_src.center;
        f_dst.center = { c_src.x(), c_src.y(), c_src.z() };

        // Face area.
        f_dst.area = f_src.area;

        // Face unit normal.
        f_dst.n10 = { f_src.n_RL.x(), f_src.n_RL.y(), f_src.n_RL.z() };
        f_dst.n01 = { f_src.n_LR.x(), f_src.n_LR.y(), f_src.n_LR.z() };

        // Face included nodes.
        const auto N1 = f_src.includedNode.size();
        f_dst.vertex.resize(N1);
        for (int j = 1; j <= N1; ++j)
            f_dst.vertex(j) = &pnt(f_src.includedNode(j));

        // Face adjacent 2 cells.
        // It should be noted that both "c0" and "c1" used in this solver
        // is DIFFERENT from that defined in Fluent Mesh Convention!!!
        // In fact, "c0" corresponds to "cr"(rightCell) and
        // "c1" corresponds to "cl"(leftCell) in Fluent Mesh.
        f_dst.c0 = f_src.leftCell ? &cell(f_src.leftCell) : nullptr;
        f_dst.c1 = f_src.rightCell ? &cell(f_src.rightCell) : nullptr;
    }

    /// Update cell information.
    for (int i = 1; i <= NumOfCell; ++i)
    {
        const auto &c_src = mesh.cell(i);
        auto &c_dst = cell(i);

        // Assign cell index.
        c_dst.index = i;

        // Cell center location.
        const auto &centroid_src = c_src.center;
        c_dst.center = { centroid_src.x(), centroid_src.y(), centroid_src.z() };

        // Cell volume.
        c_dst.volume = c_src.volume;

        // Cell included nodes.
        const auto N1 = c_src.includedNode.size();
        c_dst.vertex.resize(N1);
        for (int j = 1; j <= N1; ++j)
            c_dst.vertex(j) = &pnt(c_src.includedNode(j));

        // Cell included faces.
        const auto N2 = c_src.includedFace.size();
        c_dst.surface.resize(N2);
        for (int j = 1; j <= N2; ++j)
        {
            const auto cfi = c_src.includedFace(j);
            c_dst.surface(j) = &face(cfi);
        }

        // Cell adjacent cells.
        c_dst.adjCell.resize(N2);
        for (int j = 1; j <= N2; ++j)
        {
            const auto adjIdx = c_src.adjacentCell(j);
            c_dst.adjCell(j) = adjIdx ? &cell(adjIdx) : nullptr;
        }

        // Cell surface vectors.
        c_dst.S.resize(N2);
        for (int j = 0; j < N2; ++j)
        {
            const auto &csv = c_src.S.at(j);
            c_dst.S.at(j) = { csv.x(), csv.y(), csv.z() };
        }
    }

    /// Nodal interpolation coefficients.
    for (int i = 1; i <= NumOfPnt; ++i)
    {
        auto &n_dst = pnt(i);
        Scalar s = 0.0;
        for (int j = 1; j <= n_dst.cellWeightingCoef.size(); ++j)
        {
            auto curAdjCell = n_dst.dependentCell(j);
            const Scalar coef = 1.0 / (n_dst.coordinate - curAdjCell->center).norm();
            n_dst.cellWeightingCoef(j) = coef;
            s += coef;
        }
        for (int j = 1; j <= n_dst.cellWeightingCoef.size(); ++j)
            n_dst.cellWeightingCoef(j) /= s;
    }

    /// Cell center to face center vectors and ratios.
    for (int i = 1; i <= NumOfFace; ++i)
    {
        auto &f_dst = face(i);

        /// Displacement vectors.
        if (f_dst.c0)
            f_dst.r0 = f_dst.center - f_dst.c0->center;
        else
            f_dst.r0 = ZERO_VECTOR;

        if (f_dst.c1)
            f_dst.r1 = f_dst.center - f_dst.c1->center;
        else
            f_dst.r1 = ZERO_VECTOR;

        /// Displacement ratios.
        if (f_dst.atBdry)
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

    /// Count valid patches.
    size_t NumOfPatch = 0;
    for (int i = 1; i <= mesh.numOfZone(); ++i)
    {
        const auto &z_src = mesh.zone(i);

        const auto curZoneIdx = z_src.ID;
        auto curZonePtr = z_src.obj;
        auto curFace = dynamic_cast<XF::FACE*>(curZonePtr);
        if (curFace)
        {
            if (curFace->identity() != XF::SECTION::FACE || curFace->zone() != curZoneIdx)
                throw std::runtime_error("Inconsistency detected.");

            if (XF::BC::str2idx(z_src.type) != XF::BC::INTERIOR)
                ++NumOfPatch;
        }
    }

    /// Update boundary patch information.
    patch.resize(NumOfPatch);
    for (int i = 1, cnt = 0; i <= mesh.numOfZone(); ++i)
    {
        const auto &curZone = mesh.zone(i);
        auto curFace = dynamic_cast<XF::FACE*>(curZone.obj);
        if (curFace == nullptr || XF::BC::str2idx(curZone.type) == XF::BC::INTERIOR)
            continue;

        auto &p_dst = patch[cnt];
        p_dst.name = curZone.name;
        p_dst.BC = XF::BC::str2idx(curZone.type);
        p_dst.surface.resize(curFace->num());
        const auto loc_first = curFace->first_index();
        const auto loc_last = curFace->last_index();
        for (auto j = loc_first; j <= loc_last; ++j)
        {
            p_dst.surface.at(j - loc_first) = &face(j);
            face(j).parent = &p_dst;
        }
        cnt += 1;
    }

    /// Update nodal inclusion within each boundary patch.
    for(auto &p : patch)
    {
        const auto &s = p.surface;

        /// Counting nodes.
        std::map<int, size_t> n2n;
        size_t cnt = 0;
        for(auto f : s)
        {
            const auto &vl = f->vertex;
            for(auto v : vl)
            {
                const int ci = v->index;
                auto it = n2n.find(ci);
                if(it == n2n.end())
                {
                    ++cnt;
                    n2n[ci] = cnt;
                }
            }
        }

        /// Allocate storage.
        p.vertex.resize(n2n.size(), nullptr);

        /// Link connectivity.
        std::vector<bool> flag(n2n.size(), false);
        for(auto f : s)
        {
            const auto &vl = f->vertex;
            for(auto v : vl)
            {
                const int ci = v->index;
                auto it = n2n.find(ci);
                if(it == n2n.end())
                    throw std::runtime_error("Previous operation failed.");
                else
                {
                    const auto loc_idx = it->second - 1; /// 0-based
                    if(!flag[loc_idx])
                    {
                        p.vertex[loc_idx] = v;
                        flag[loc_idx] = true;
                    }
                }
            }
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
                cur_d = cur_face->center - cur_cell.center;
            else
                cur_d = cur_adj_cell->center - cur_cell.center;

            // Non-Orthogonal correction
            calc_non_orthogonal_correction_vector(NOC_Method, cur_d, cur_cell.S[j], cur_cell.Se[j], cur_cell.St[j]);
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
