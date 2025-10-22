#ifndef INITIAL_FUNCTION_FACTORY_HPP
#define INITIAL_FUNCTION_FACTORY_HPP

#include "Globals.hpp"

#include <cmath>
#include <Eigen/Dense>
#include <utility>
#include <vector>
namespace InitialFunctionFactory
{
    Eigen::VectorXd cosHill(
        const int    nx, const int    ny,
        const double dx, const double dy
    );

    Eigen::VectorXd twinCosHill(
        const int    nx, const int    ny,
        const double dx, const double dy
    );

    Eigen::VectorXd debugData(
        const int    nx, const int    ny,
        const double dx, const double dy
    );
}

#endif //INITIAL_FUNCTION_FACTORY_HPP