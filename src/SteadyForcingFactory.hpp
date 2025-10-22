#ifndef STEADY_FORCING_FACTORY_HPP
#define STEADY_FORCING_FACTORY_HPP

#include "Globals.hpp"

#include <cmath>
#include <Eigen/Dense>
#include <iostream>
#include <utility>
#include <vector>

Eigen::VectorXd constantFunction1D(
    const int nx,
    const double forcing_amplitude
); // 1-D

Eigen::VectorXd constantFunction2D(
    const int nx, const int ny,
    const double forcing_amplitude
); // 2-D

Eigen::VectorXd sineSine(
    const int    nx, const int    ny,
    const double dx, const double dy
);

#endif //STEADY_FORCING_FACTORY_HPP