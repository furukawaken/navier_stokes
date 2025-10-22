#ifndef TRIPLET_LIST_FACTORY_HPP
#define TRIPLET_LIST_FACTORY_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <execution>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "Parameters.hpp"
#include "BoundaryConditionChecker.hpp"

class LaplacianTripletFactory {
public:
    LaplacianTripletFactory() = default;
    std::vector<Eigen::Triplet<double>> getList(
        const int    nx,  const int    ny,
        const double dx,  const double dy,
        const double nu,
        const BoundaryConditionChecker& bc_checker);
};

/* Solve heat equations under Dirichlet and Neumann b.c..
    Staggered Lattice
    Equation:
        du/dt = \Delta u + f
    Discretization:
        \Delta
        = \nu_x (u_i+1,j - 2u_i,j + u_i-1,j)/dx^2
        + \nu_y (u_i,j+1 - 2u_i,j + u_i,j-1)/dx^2
    
    Boundary conditions:
        u = g
        du/dnu = g
    Discretization:
        u_j+1/2 + u_j-1/2 = 2g_j
        u_j+1/2 - u_j-1/2 = dx * g_j
    i.e.
        u_j+1/2 = 2g_j     - u_j-1/2
        u_j+1/2 = dx * g_j + u_j-1/2
    
    Variable point types:
        Node-centered              (NC)
        Cell-centered              (CC)
        Hybrid-centered(Staggered) (HC)
*/
class InhomogeneousHeatTripletFactory {
public:
    InhomogeneousHeatTripletFactory() = default;
    std::vector<Eigen::Triplet<double>> getEvolutionMatrixList(
        const int nx, const int ny, const double dx, const double dy, const double nu, const BoundaryConditionChecker& bc_checker
    );
};

class HorizontalVelocityTripletFactory {
public:
    HorizontalVelocityTripletFactory() = default;
    std::vector<Eigen::Triplet<double>> getDirichletList(const int nx, const int ny, const double dx, const double dy, const double reynolds);
};

class VerticalVelocityTripletFactory {
public:
    VerticalVelocityTripletFactory() = default;
    std::vector<Eigen::Triplet<double>> getDirichletList(const int nx, const int ny, const double dx, const double dy, const double reynolds);
};

#endif // TRIPLET_LIST_FACTORY_HPP
