#ifndef INITIAL_VELOCITY_FACTORY_HPP
#define INITIAL_VELOCITY_FACTORY_HPP

#include "Globals.hpp"

#include <cmath>
#include <Eigen/Dense>
#include <iostream>
#include <utility>
#include <vector>
namespace InitialVelocityAssembler
{
    void taylorGreenFlow(
        const int    nx, const int    ny,
        const double dx, const double dy,
        const double reynolds,
        const double xmode, const double ymode,
        Eigen::VectorXd& u,
        Eigen::VectorXd& v,
        Eigen::VectorXd& p,
        const double t = 0.0
    );
}

#endif //INITIAL_VELOCITY_FACTORY_HPP