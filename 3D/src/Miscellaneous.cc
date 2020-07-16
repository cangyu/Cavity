#include <chrono>
#include <sstream>
#include <iomanip>
#include "../inc/Miscellaneous.h"
#include "../inc/custom_type.h"

const std::array<std::string, 3> unsupported_boundary_condition::BC_CATEGORY_STR = {
        "Dirichlet",
        "Neumann",
        "Robin"
};

double duration(const clock_t &startTime, const clock_t &endTime)
{
    return static_cast<double>(endTime - startTime) / CLOCKS_PER_SEC;
}

std::string time_stamp_str()
{
    auto tt = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::ostringstream ss;
    ss << std::put_time(std::localtime(&tt), "%Y%m%d-%H%M%S");
    return ss.str();
}
