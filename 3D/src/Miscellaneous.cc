#include <chrono>
#include <sstream>
#include <iomanip>
#include "../inc/Miscellaneous.h"

double duration(const clock_t &startTime, const clock_t &endTime)
{
    return (endTime - startTime) * 1.0 / CLOCKS_PER_SEC;
}

std::string time_stamp_str()
{
    auto tt = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::ostringstream ss;
    ss << std::put_time(std::localtime(&tt), "%Y%m%d-%H%M%S");
    return ss.str();
}
