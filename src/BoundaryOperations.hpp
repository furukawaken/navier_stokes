#ifndef BOUNDARY_OPERATIONS_HPP
#define BOUNDARY_OPERATIONS_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <execution>
#include <iostream>
#include <vector>

#include "Parameters.hpp"
#include "BoundaryConditionChecker.hpp"

namespace Extensions
{
    // u must have Dirichlet bc for left ant right
    Eigen::VectorXd getExtendedULeftRight(
        const Eigen::VectorXd& u,
        const Eigen::VectorXd& left,
        const Eigen::VectorXd& right,
        const int    nx,
        const int    ny,
        const double dx
    );

    // v must have Dirichlet bc for top ant bottom
    Eigen::VectorXd getExtendedVTopBottom(
        const Eigen::VectorXd& v,
        const Eigen::VectorXd& top,
        const Eigen::VectorXd& bottom,
        const int    nx,
        const int    ny,
        const double dy
    );
} // namespace Extensions

namespace BoundaryValueHandler
{
    void setNoSlip(
        std::vector<Eigen::VectorXd>& uv,
        const int nxu, const int nyu,
        const int nxv, const int nyv
    );
} // namespace BoundaryValueHandler

#endif // BOUNDARY_OPERATIONS_HPP
