#include <cmath>
#include <cstdio>
#include <fstream>
#include <execution>
#include <Eigen/Sparse>
#include <iostream>
#include <memory>
#include <netcdf.h>
#include <stdexcept>
#include <vector>
#include <utility>
#include <omp.h>
#include <chrono>

// My Class Files
#include "../src/BoundaryConditionChecker.hpp"
#include "../src/BoundaryForcingBuilderCellCentered.hpp"
#include "../src/BCModificationToolsNeumann.hpp"
#include "../src/DataWriter.hpp"
#include "../src/Globals.hpp"
#include "../src/Parameters.hpp"
#include "../src/ParameterConverter.hpp"
#include "../src/SteadyForcingFactory.hpp"
#include "../src/TripletListFactory.hpp"

void testLinearPoissonDirichlet() {
    std::cout << "testLinearPoissonDirichletByCGM()" << std::endl;

    // Numerical settings
    Parameters parameters;
    int    nx   = 100;
    int    ny   = 100;
    double dx   = 1.0;// / nx;
    double dy   = 1.0;// / ny;
    double nu = parameters.diffusivity;

    // Setting parameters
    parameters.nx = nx;
    parameters.ny = ny;
    parameters.dx = dx;
    parameters.dy = dy;

    // Setting a boundary condition
    BoundaryConditionChecker bc_checker("D");

    // Setting a writer
    std::string filename{"../output/test1/poisson_dirichlet_cgm.nc"};
    NetCDFWriter writer(
        filename,
        0.0,      // dummy time
        dy, dx,
        ny, nx
    );
    writer.createFile(); // ncfile creation

    // Setting vectors
    Eigen::VectorXd u
        = Eigen::VectorXd::Zero(nx * ny);
    Eigen::VectorXd total_forcing
        = Eigen::VectorXd::Zero(nx * ny)
        + BoundaryForcingBuilderCellCentered(
            nx,   ny,
            dx,   dy,
            nu,
            bc_checker
        ).getBoundaryForcingFrom(
            Eigen::VectorXd::Ones(nx),
            Eigen::VectorXd::Ones(nx),
            Eigen::VectorXd::Zero(ny),
            Eigen::VectorXd::Zero(ny)
        );

    // Evolution matrix settings
    Eigen::SparseMatrix<double> matrix(nx * ny, nx * ny);;
    std::vector<Eigen::Triplet<double>> triplet_list
        = LaplacianTripletFactory().getList(
            nx,   ny,
            dx,   dy,
            nu,
            bc_checker
        );
    
    // Making a solver
    matrix.setFromTriplets(triplet_list.begin(), triplet_list.end());
    matrix.makeCompressed();
    Eigen::ConjugateGradient<
        Eigen::SparseMatrix<double>,
        Eigen::Lower | Eigen::Upper,
        Eigen::IncompleteCholesky<double>
    > solver;
    solver.compute(matrix);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("IncompleteCholesky factorization failed!");
    }

    // Applying the solver
    //u = lscg.solve(total_forcing);
    auto t1 = std::chrono::high_resolution_clock::now();
    u = solver.solve(total_forcing);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = t2 - t1;
    std::cout << "Initial Poisson solve took " << elapsed.count() << " seconds" << std::endl;
    
    // If average-free is required:
    //u = u - u.mean() * Eigen::VectorXd::Ones(nx * ny);

    // Save results:
    std::vector<double> std_vec(u.data(), u.data() + u.size());
    writer.writeData(std_vec);
}

void testLinearPoissonNeumann() {
    std::cout << "testLinearPoissonNeumannByCGM()" << std::endl;

    // Numerical settings
    Parameters parameters;
    int    nx   = 100;
    int    ny   = 100;
    double dx   = 0.1;// / nx;
    double dy   = 0.1;// / ny;
    double nu = parameters.diffusivity;

    // Setting parameters
    parameters.nx = nx;
    parameters.ny = ny;
    parameters.dx = dx;
    parameters.dy = dy;

    // Setting a boundary condition
    BoundaryConditionChecker bc_checker("N");

    // Setting a writer
    std::string filename{"../output/test2/poisson_neumann_llt.nc"};
    NetCDFWriter writer(
        filename,
        0.0,      // dummy time
        dy, dx,
        ny, nx
    );
    writer.createFile(); // ncfile creation

    // Setting vectors
    Eigen::VectorXd u
        = sineSine(nx, ny, dx, dy);
    Eigen::VectorXd total_forcing
        = Eigen::VectorXd::Zero(nx * ny)
        + BoundaryForcingBuilderCellCentered(
            nx,   ny,
            dx,   dy,
            nu,
            bc_checker
        ).getBoundaryForcingFrom(
            Eigen::VectorXd::Zero(nx),
            Eigen::VectorXd::Zero(nx),
            Eigen::VectorXd::Ones(ny),
            Eigen::VectorXd::Ones(ny)
        );

    // Making a solver
    std::vector<Eigen::Triplet<double>> triplet_list
        = LaplacianTripletFactory().getList(
            nx,   ny,
            dx,   dy,
            nu,
            bc_checker
        );
    Eigen::SparseMatrix<double> matrix(nx * ny, nx * ny);
    matrix.setFromTriplets(triplet_list.begin(), triplet_list.end());
    BCModificationToolsNeumann::pinMatrix(matrix, nx, ny);
    matrix.makeCompressed();
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(matrix);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("SimplicialLLT factorization failed");
    }
    /*
    Eigen::ConjugateGradient<
        Eigen::SparseMatrix<double>,
        Eigen::Lower | Eigen::Upper,
        Eigen::IncompleteCholesky<double>
    > pcg;
    pcg.compute(positive_laplacian);
    if (pcg.info() != Eigen::Success) {
        throw std::runtime_error("IncompleteCholesky factorization failed!");
    }
    */

    // Applying the solver
    //u = lscg.solve(total_forcing);
    auto t1 = std::chrono::high_resolution_clock::now();
    u = solver.solve(total_forcing);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = t2 - t1;
    std::cout << "Initial Poisson solve took " << elapsed.count() << " seconds" << std::endl;
    
    // If average-free is required:
    u = u - u.mean() * Eigen::VectorXd::Ones(nx * ny);

    // Save results:
    std::vector<double> std_vec(u.data(), u.data() + u.size());
    writer.writeData(std_vec);
}
