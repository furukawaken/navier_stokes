#ifndef MATH_FUNCTIONS_HPP
#define MATH_FUNCTIONS_HPP

#include <cmath>
#include <Eigen/Dense>
#include <iostream>
#include <utility>
#include <vector>
#include "Globals.hpp"

namespace MathFunctions{
    std::vector<Eigen::VectorXd> oseenVortex(
        const int nx, const int ny,
        const double dx, const double dy,
        const double scale, double circulation
    );

    std::vector<Eigen::VectorXd> oseenVortexCentered(
        const int nx, const int ny,
        const double dx, const double dy,
        const double scale, double circulation
    );
};

#endif //MATH_FUNCTIONS_HPP