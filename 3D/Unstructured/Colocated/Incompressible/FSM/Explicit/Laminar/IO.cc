#include <fstream>
#include "xf.h"
#include "custom_type.h"
#include "IO.h"

extern size_t NumOfPnt;
extern size_t NumOfFace;
extern size_t NumOfCell;

extern NaturalArray<Point> pnt; // Node objects
extern NaturalArray<Face> face; // Face objects
extern NaturalArray<Cell> cell; // Cell objects
extern NaturalArray<Patch> patch; // Group of boundary faces

/*************************************************** File I/O ********************************************************/

/**
 * Load mesh.
 * @param MESH_PATH
 */
void readMESH(const std::string &MESH_PATH)
{
    using namespace GridTool;

    // Load ANSYS Fluent mesh using external independent package.
    const XF::MESH mesh(MESH_PATH);

    // Update counting of geom elements.
    NumOfPnt = mesh.numOfNode();
    NumOfFace = mesh.numOfFace();
    NumOfCell = mesh.numOfCell();

    // Allocate memory for geom information and related physical variables.
    pnt.resize(NumOfPnt);
    face.resize(NumOfFace);
    cell.resize(NumOfCell);

    // Update point information.
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

    // Update face information.
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
        f_dst.c0 = f_src.leftCell ? &cell(f_src.leftCell) : nullptr;
        f_dst.c1 = f_src.rightCell ? &cell(f_src.rightCell) : nullptr;
    }

    // Update cell information.
    for (int i = 1; i <= NumOfCell; ++i)
    {
        const auto &c_src = mesh.cell(i);
        auto &c_dst = cell(i);

        // Assign cell index.
        c_dst.index = i;

        // Cell center location.
        const auto &centroid_src = c_src.center;
        c_dst.center = { centroid_src.x(), centroid_src.y(), centroid_src.z() };

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

    // Nodal interpolation coefficients
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

    // Cell center to face center vectors 
    for (int i = 1; i <= NumOfFace; ++i)
    {
        auto &f_dst = face(i);

        if (f_dst.c0)
            f_dst.r0 = f_dst.center - f_dst.c0->center;
        else
            f_dst.r0 = ZERO_VECTOR;

        if (f_dst.c1)
            f_dst.r1 = f_dst.center - f_dst.c1->center;
        else
            f_dst.r1 = ZERO_VECTOR;
    }

    // Count valid patches.
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

    // Update boundary patch information.
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
}

/**
 * Output computation results.
 * Nodal values are exported, including boundary variables.
 * @param fn
 * @param title
 */
void writeTECPLOT_Nodal(const std::string &fn, const std::string &title)
{
    static const size_t RECORD_PER_LINE = 10;

    std::ofstream fout(fn);
    if (fout.fail())
        throw failed_to_open_file(fn);

    fout << R"(TITLE=")" << title << "\"" << std::endl;
    fout << "FILETYPE=FULL" << std::endl;
    fout << R"(VARIABLES="X", "Y", "Z", "rho", "U", "V", "W", "P", "T")" << std::endl;

    fout << R"(ZONE T="Nodal")" << std::endl;
    fout << "NODES=" << NumOfPnt << ", ELEMENTS=" << NumOfCell << ", ZONETYPE=FEBRICK, DATAPACKING=BLOCK, VARLOCATION=([1-9]=NODAL)" << std::endl;

    // X-Coordinates
    for (size_t i = 1; i <= NumOfPnt; ++i)
    {
        fout << '\t' << pnt(i).coordinate.x();
        if (i % RECORD_PER_LINE == 0)
            fout << std::endl;
    }
    if (NumOfPnt % RECORD_PER_LINE != 0)
        fout << std::endl;

    // Y-Coordinates
    for (size_t i = 1; i <= NumOfPnt; ++i)
    {
        fout << '\t' << pnt(i).coordinate.y();
        if (i % RECORD_PER_LINE == 0)
            fout << std::endl;
    }
    if (NumOfPnt % RECORD_PER_LINE != 0)
        fout << std::endl;

    // Z-Coordinates
    for (size_t i = 1; i <= NumOfPnt; ++i)
    {
        fout << '\t' << pnt(i).coordinate.z();
        if (i % RECORD_PER_LINE == 0)
            fout << std::endl;
    }
    if (NumOfPnt % RECORD_PER_LINE != 0)
        fout << std::endl;

    // Density
    for (size_t i = 1; i <= NumOfPnt; ++i)
    {
        fout << '\t' << pnt(i).rho;
        if (i % RECORD_PER_LINE == 0)
            fout << std::endl;
    }
    if (NumOfPnt % RECORD_PER_LINE != 0)
        fout << std::endl;

    // Velocity-X
    for (size_t i = 1; i <= NumOfPnt; ++i)
    {
        fout << '\t' << pnt(i).U.x();
        if (i % RECORD_PER_LINE == 0)
            fout << std::endl;
    }
    if (NumOfPnt % RECORD_PER_LINE != 0)
        fout << std::endl;

    // Velocity-Y
    for (size_t i = 1; i <= NumOfPnt; ++i)
    {
        fout << '\t' << pnt(i).U.y();
        if (i % RECORD_PER_LINE == 0)
            fout << std::endl;
    }
    if (NumOfPnt % RECORD_PER_LINE != 0)
        fout << std::endl;

    // Velocity-Z
    for (size_t i = 1; i <= NumOfPnt; ++i)
    {
        fout << '\t' << pnt(i).U.z();
        if (i % RECORD_PER_LINE == 0)
            fout << std::endl;
    }
    if (NumOfPnt % RECORD_PER_LINE != 0)
        fout << std::endl;

    // Pressure
    for (size_t i = 1; i <= NumOfPnt; ++i)
    {
        fout << '\t' << pnt(i).p;
        if (i % RECORD_PER_LINE == 0)
            fout << std::endl;
    }
    if (NumOfPnt % RECORD_PER_LINE != 0)
        fout << std::endl;

    // Temperature
    for (size_t i = 1; i <= NumOfPnt; ++i)
    {
        fout << '\t' << pnt(i).T;
        if (i % RECORD_PER_LINE == 0)
            fout << std::endl;
    }
    if (NumOfPnt % RECORD_PER_LINE != 0)
        fout << std::endl;

    // Connectivity Information
    for (size_t i = 1; i <= NumOfCell; ++i)
    {
        const auto v = cell(i).vertex;
        for (const auto &e : v)
            fout << '\t' << e->index;
        fout << std::endl;
    }
    if (NumOfCell % RECORD_PER_LINE != 0)
        fout << std::endl;

    fout.close();
}

/**
 * Output computation results.
 * Cell-Centered values are exported, boundary variables are NOT included.
 * Only for continuation purpose.
 * @param fn
 * @param title
 */
void writeTECPLOT_CellCentered(const std::string &fn, const std::string &title)
{
    static const size_t RECORD_PER_LINE = 10;

    std::ofstream fout(fn);
    if (fout.fail())
        throw failed_to_open_file(fn);

    fout << R"(TITLE=")" << title << "\"" << std::endl;
    fout << "FILETYPE=FULL" << std::endl;
    fout << R"(VARIABLES="X", "Y", "Z", "rho", "U", "V", "W", "P", "T")" << std::endl;

    fout << R"(ZONE T="Cell-Centroid")" << std::endl;
    fout << "NODES=" << NumOfPnt << ", ELEMENTS=" << NumOfCell << ", ZONETYPE=FEBRICK, DATAPACKING=BLOCK, VARLOCATION=([1-3]=NODAL, [4-9]=CELLCENTERED)" << std::endl;

    // X-Coordinates
    for (size_t i = 1; i <= NumOfPnt; ++i)
    {
        fout << '\t' << pnt(i).coordinate.x();
        if (i % RECORD_PER_LINE == 0)
            fout << std::endl;
    }
    if (NumOfPnt % RECORD_PER_LINE != 0)
        fout << std::endl;

    // Y-Coordinates
    for (size_t i = 1; i <= NumOfPnt; ++i)
    {
        fout << '\t' << pnt(i).coordinate.y();
        if (i % RECORD_PER_LINE == 0)
            fout << std::endl;
    }
    if (NumOfPnt % RECORD_PER_LINE != 0)
        fout << std::endl;

    // Z-Coordinates
    for (size_t i = 1; i <= NumOfPnt; ++i)
    {
        fout << '\t' << pnt(i).coordinate.z();
        if (i % RECORD_PER_LINE == 0)
            fout << std::endl;
    }
    if (NumOfPnt % RECORD_PER_LINE != 0)
        fout << std::endl;

    // Density
    for (size_t i = 1; i <= NumOfCell; ++i)
    {
        fout << '\t' << cell(i).rho0;
        if (i % RECORD_PER_LINE == 0)
            fout << std::endl;
    }
    if (NumOfCell % RECORD_PER_LINE != 0)
        fout << std::endl;

    // Velocity-X
    for (size_t i = 1; i <= NumOfCell; ++i)
    {
        fout << '\t' << cell(i).U0.x();
        if (i % RECORD_PER_LINE == 0)
            fout << std::endl;
    }
    if (NumOfCell % RECORD_PER_LINE != 0)
        fout << std::endl;

    // Velocity-Y
    for (size_t i = 1; i <= NumOfCell; ++i)
    {
        fout << '\t' << cell(i).U0.y();
        if (i % RECORD_PER_LINE == 0)
            fout << std::endl;
    }
    if (NumOfCell % RECORD_PER_LINE != 0)
        fout << std::endl;

    // Velocity-Z
    for (size_t i = 1; i <= NumOfCell; ++i)
    {
        fout << '\t' << cell(i).U0.z();
        if (i % RECORD_PER_LINE == 0)
            fout << std::endl;
    }
    if (NumOfCell % RECORD_PER_LINE != 0)
        fout << std::endl;

    // Pressure
    for (size_t i = 1; i <= NumOfCell; ++i)
    {
        fout << '\t' << cell(i).p0;
        if (i % RECORD_PER_LINE == 0)
            fout << std::endl;
    }
    if (NumOfCell % RECORD_PER_LINE != 0)
        fout << std::endl;

    // Temperature
    for (size_t i = 1; i <= NumOfCell; ++i)
    {
        fout << '\t' << cell(i).T0;
        if (i % RECORD_PER_LINE == 0)
            fout << std::endl;
    }
    if (NumOfCell % RECORD_PER_LINE != 0)
        fout << std::endl;

    // Connectivity Information
    for (size_t i = 1; i <= NumOfCell; ++i)
    {
        const auto v = cell(i).vertex;
        for (const auto &e : v)
            fout << '\t' << e->index;
        fout << std::endl;
    }
    if (NumOfCell % RECORD_PER_LINE != 0)
        fout << std::endl;

    fout.close();
}

/**
 * Extract NODAL solution variables only.
 * Should be consistent with existing mesh!!!
 * @param fn
 */
void readTECPLOT_Nodal(const std::string &fn)
{
    std::ifstream fin(fn);
    if (fin.fail())
        throw failed_to_open_file(fn);

    /* Skip header */
    for (int i = 0; i < 5; ++i)
    {
        std::string tmp;
        std::getline(fin, tmp);
    }

    /* Skip coordinates */
    for (int k = 0; k < 3; ++k)
    {
        Scalar var;
        for (size_t i = 1; i <= NumOfPnt; ++i)
            fin >> var;
    }

    /* Load data */
    // Density
    for (size_t i = 1; i <= NumOfPnt; ++i)
        fin >> pnt(i).rho;

    // Velocity-X
    for (size_t i = 1; i <= NumOfPnt; ++i)
        fin >> pnt(i).U.x();

    // Velocity-Y
    for (size_t i = 1; i <= NumOfPnt; ++i)
        fin >> pnt(i).U.y();

    // Velocity-Z
    for (size_t i = 1; i <= NumOfPnt; ++i)
        fin >> pnt(i).U.z();

    // Pressure
    for (size_t i = 1; i <= NumOfPnt; ++i)
        fin >> pnt(i).p;

    // Temperature
    for (size_t i = 1; i <= NumOfPnt; ++i)
        fin >> pnt(i).T;

    /* Finalize */
    fin.close();
}

/**
 * Extract CELL-CENTERED solution variables only.
 * Should be consistent with existing mesh!!!
 * Only for continuation purpose.
 * @param fn
 */
void readTECPLOT_CellCentered(const std::string &fn)
{
    std::ifstream fin(fn);
    if (fin.fail())
        throw failed_to_open_file(fn);

    /* Skip header */
    for (int i = 0; i < 5; ++i)
    {
        std::string tmp;
        std::getline(fin, tmp);
    }

    /* Skip coordinates */
    for (int k = 0; k < 3; ++k)
    {
        Scalar var;
        for (size_t i = 1; i <= NumOfPnt; ++i)
            fin >> var;
    }

    /* Load data */
    // Density
    for (size_t i = 1; i <= NumOfCell; ++i)
        fin >> cell(i).rho0;

    // Velocity-X
    for (size_t i = 1; i <= NumOfCell; ++i)
        fin >> cell(i).U0.x();

    // Velocity-Y
    for (size_t i = 1; i <= NumOfCell; ++i)
        fin >> cell(i).U0.y();

    // Velocity-Z
    for (size_t i = 1; i <= NumOfCell; ++i)
        fin >> cell(i).U0.z();

    // Pressure
    for (size_t i = 1; i <= NumOfCell; ++i)
        fin >> cell(i).p0;

    // Temperature
    for (size_t i = 1; i <= NumOfCell; ++i)
        fin >> cell(i).T0;

    /* Finalize */
    fin.close();
}
