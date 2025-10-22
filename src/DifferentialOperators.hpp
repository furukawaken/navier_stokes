#ifndef DIFFERENTIAL_OPERATORS_HPP
#define DIFFERENTIAL_OPERATORS_HPP

#include <cmath>
#include <execution>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <stdexcept>
#include <vector>

// For calculus for the pressure
/*
    Recall
    Grid numbers sheet:(phi = scalar field)
    phi : (xi_phi, eta_phi)
      u : (  xi_u,   eta_u) = (xi_phi - 1, eta_phi)
      v : (  xi_v,   eta_v) = (xi_phi,     eta_rho - 1)
*/

namespace DifferentialOperators {
    /*
      v     v            v     v
    u phi u phi u ---- u phi u phi u
      v     v            v     v
    u phi u phi u ---- u phi u phi u
      v     v            v     v
      |     |            |     |
      |     |            |     |
      v     v            v     v
    u phi u phi u ---- u phi u phi u
      v     v            v     v
    u phi u phi u ---- u phi u phi u
      v     v            v     v
    
    u have value on the left and right boundaries
    v have value on the top and bottom boundaries
    */
    std::vector<Eigen::VectorXd> nablaOfScalerField(
        const Eigen::VectorXd& p,
        const int    nx,
        const int    ny,
        const double dx,
        const double dy
    );

	Eigen::VectorXd divergenceOfVectorField(
            const Eigen::VectorXd& u_extended_to_left_right,
            const Eigen::VectorXd& v_extended_to_top_bottom,
            const int    nx,
            const int    ny,
            const double dx,
            const double dy
    );
};
/* Upwind Difference
    u df/dx
    = u [ - f_i+2 
          + 8 ( f_i+1 - f_i-1 )
          + f_i-2
        ]
    + ( alpha * |u| * dx^3 / 12 ) * [
        f_i+2
        - 4 f_i+1
        + 6 f_i
        - 4 f_i-1
        + f_i-2
    ] / dx^4
*/

#endif // DIFFERENTIAL_OPERATORS_HPP