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
#include "../src/IndexMapping.hpp"
#include "../src/InitialFunctionFactory.hpp"
#include "../src/MathFunctions.hpp"
#include "../src/Parameters.hpp"
#include "../src/SteadyForcingFactory.hpp"
#include "../src/TripletListFactory.hpp"

/*
void pinMatrix(Eigen::SparseMatrix<double>& laplacian, const int nx, const int ny) {
    laplacian.coeffRef(0, 0) = 1.0;
    for (int i = 1; i < nx * ny; ++i) {
        laplacian.coeffRef(0, i) = 0.0;
        laplacian.coeffRef(i, 0) = 0.0;
    }
}

Eigen::VectorXd getPinedVector(const Eigen::VectorXd& u) {
    Eigen::VectorXd u_pinned(u);
    u_pinned[0] = 0.0;
    return u_pinned;
}
*/

// Compute chemotactic flux divergence term: - div (u nabla v)
Eigen::VectorXd computeChemotaxis(
    const Eigen::VectorXd& u,
    const Eigen::VectorXd& v,
    const int    nx, const int    ny,
    const double dx, const double dy,
    const double chi=1.0
) {
    // Parameters
    int    center{0};
    double djx_dx;
    double djy_dy;
    //double chi{1.0};

    Eigen::VectorXd flux = Eigen::VectorXd::Zero(u.size());

    auto index = [nx](int ix, int iy) { return iy * nx + ix; };
    
    #pragma omp parallel for collapse(1) schedule(static)
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            center = index(i, j);

            int i_plus  = (i < nx - 1) ? i + 1 : i;
            int i_minus = (i > 0     ) ? i - 1 : i;
            int j_plus  = (j < ny - 1) ? j + 1 : j;
            int j_minus = (j > 0     ) ? j - 1 : j;

            // Jx = -chi * u * dv/dx
            djx_dx = - chi * (
                (
                      u[index(i_plus, j)]
                    + u[center]
                )
                * (
                      v[index(i_plus, j)]
                    - v[center]
                )
                - (
                      u[center]
                    + u[index(i_minus, j)]
                )
                * (
                      v[center]
                    - v[index(i_minus, j)]
                )
            ) / (dx * dx);

            // Jy = -chi * u * dv/dy
            djy_dy = - chi * (
                (
                      u[index(i, j_plus)]
                    + u[index(i, j    )]
                )
                * (
                      v[index(i, j_plus)]
                    - v[center]
                )
                - (
                      u[center]
                    + u[index(i, j_minus)]
                )
                * (
                      v[center]
                    - v[index(i, j_minus)]
                )
            ) / (dy * dy);

            flux[center] = djx_dx + djy_dy;
        }
    }

    return flux;
}

void testKellerSegelNeumann() {
    std::cout << "testKellerSegelNeumann" << std::endl;

    // Saving settings
    int max_writing_count{20};
    int writing_count{0};
    double t{0.0};
    bool shouldStop = false;

    // Parameters settings
    Parameters parameters;
    int    si   = 10;
    int    nt   = max_writing_count * si;
    int    nx   = 250;
    int    ny   = 250;
    double dt   = 0.0000001;
    double dx   = 0.001;
    double dy   = 0.001;
    double nu = 1.0;
    double chemotaxis_chi = 1.0;

    // Inserting parameters
    parameters.nt              = nt;
    parameters.nx              = nx;
    parameters.ny              = ny;
    parameters.dt              = dt;
    parameters.dx              = dx;
    parameters.dy              = dy;
    parameters.diffusivity     = nu;
    //parameters.diffusivity_y   = nu_y;
    parameters.saving_interval = si;

    // Setting a boundary condition
    BoundaryConditionChecker bc_checker("N");

    // Setting a writer
    std::string filename{"../output/test4/ks.nc"};
    NetCDFWriterTwoScalerFields writer(
        filename,
        dt, dy, dx,
            ny, nx
    );
    writer.createFile();
    writer.setCoordinate();

    // Setting vectors
    Eigen::VectorXd u
        //= (8.0 * PI) * InitialFunctionFactory::cosHill(nx, ny, dx, dy); // Initial data
        = 0.8 * (8.0 * PI) * InitialFunctionFactory::cosHill(nx, ny, dx, dy); // Initial data
        //= InitialFunctionFactory::twinCosHill(nx, ny, dx, dy);                // Initial data
    Eigen::VectorXd v
        = Eigen::VectorXd::Zero(nx * ny);
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
            Eigen::VectorXd::Zero(ny),
            Eigen::VectorXd::Zero(ny)
        );
    std::cout << "u0.mean() = " << u.mean() << std::endl;
    
    // Setting RK4 vectors
    Eigen::VectorXd k1
        = Eigen::VectorXd::Zero(nx * ny);
    Eigen::VectorXd k2
        = Eigen::VectorXd::Zero(nx * ny);
    Eigen::VectorXd k3
        = Eigen::VectorXd::Zero(nx * ny);
    Eigen::VectorXd k4
        = Eigen::VectorXd::Zero(nx * ny);
    Eigen::VectorXd u_clone
        = Eigen::VectorXd::Zero(nx * ny);
    // Making a linear solvers
    // Making matrices
    std::vector<Eigen::Triplet<double>> triplet_list =
        LaplacianTripletFactory().getList(
            nx, ny, dx, dy, nu, bc_checker
        ); // Positive Laplacian
    Eigen::SparseMatrix<double> positive_laplacian(nx * ny, nx * ny);
    positive_laplacian.setFromTriplets(triplet_list.begin(), triplet_list.end());
    Eigen::SparseMatrix<double> pinned_positive_laplacian(positive_laplacian);
    BCModificationToolsNeumann::pinMatrix(pinned_positive_laplacian, nx, ny);
    positive_laplacian.makeCompressed();
    pinned_positive_laplacian.makeCompressed();
    // Making a solver
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(pinned_positive_laplacian);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("SimplicialLLT factorization failed");
    }
    /* // Other option
    Eigen::ConjugateGradient<
        Eigen::SparseMatrix<double>,
        Eigen::Lower | Eigen::Upper,
        Eigen::IncompleteCholesky<double>
    > solver;
    solver.compute(positive_laplacian);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("IncompleteCholesky factorization failed!");
    }
    */

    // Solving v
    auto t1 = std::chrono::high_resolution_clock::now();
    v = solver.solve(
        BCModificationToolsNeumann::getPinedVector(
            u
            - u.mean() * Eigen::VectorXd::Ones(nx * ny)
            // sum (u[j] * dx) / dx * nx = u.mean()
        )
    );
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = t2 - t1;
    std::cout << "Initial Poisson solve took " << elapsed.count() << " seconds" << std::endl;

    // Saving initial condition
    writer.writeTime(t, writing_count);
    writer.writeVectorField(
        { std::vector<double>(u.data(), u.data() + u.size()),
          std::vector<double>(v.data(), v.data() + v.size()) },
        writing_count
    );
    writing_count++; 

    // Time integration by RK4
    for ([[maybe_unused]] int i = 0; i < max_writing_count && !shouldStop; ++i) {
        std::cout << "Integration count : " << i << std::endl;
        for (int it = 0; it < si; ++it) {
            //shouldStop = true; break;
            u_clone = u;

            // Calculations of k1:
            k1 = - positive_laplacian * u
               + computeChemotaxis(u, v, nx, ny, dx, dy, chemotaxis_chi);

            // Calculations of k2:
            u  = u_clone + 0.5 * dt * k1;
            v  = solver.solve(
                BCModificationToolsNeumann::getPinedVector(
                    u
                    - u.mean() * Eigen::VectorXd::Ones(nx * ny)
                    // sum (u[j] * dx) / dx * nx = u.mean()
                )
            );
            k2 = - positive_laplacian * u
               + computeChemotaxis(u, v, nx, ny, dx, dy, chemotaxis_chi);

            // Calculations of k3:
            u  = u_clone + 0.5 * dt * k2;
            v  = solver.solve(
                BCModificationToolsNeumann::getPinedVector(
                    u
                    - u.mean() * Eigen::VectorXd::Ones(nx * ny)
                    // sum (u[j] * dx) / dx * nx = u.mean()
                )
            );
            k3 = - positive_laplacian * u
               + computeChemotaxis(u, v, nx, ny, dx, dy, chemotaxis_chi);

            // Calculations of k4:
            u  = u_clone + dt * k3;
            v  = solver.solve(
                BCModificationToolsNeumann::getPinedVector(
                    u
                    - u.mean() * Eigen::VectorXd::Ones(nx * ny)
                )
            );
            k4 = - positive_laplacian * u
               + computeChemotaxis(u, v, nx, ny, dx, dy, chemotaxis_chi);

            // Update u, v, t:
            u = u_clone + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
            v = solver.solve(
                BCModificationToolsNeumann::getPinedVector(
                    u
                    - u.mean() * Eigen::VectorXd::Ones(nx * ny)
                )
            );
            t += dt;

            // Break if it blows up.
            if (u.maxCoeff() > 100000) {
                std::cout << "Solution is blowing up at t = " << t << std::endl;
                std::cout << "u.maxCoeff() = " << u.maxCoeff() << std::endl;
                shouldStop = true;
                break; // break inner loop
            }
            std::cout << "u.maxCoeff() = " << u.maxCoeff() << std::endl;
        }

        // Saving calculations
        writer.writeTime(t, writing_count);
        writer.writeVectorField(
            { std::vector<double>(u.data(), u.data() + u.size()),
              std::vector<double>(v.data(), v.data() + v.size()) },
            writing_count
        );
        std::cout << "u.mean() = " << u.mean() << std::endl;
        writing_count++;

        // Break if it converge to the steady state.
        double rmse = (u - u_clone).norm() / std::sqrt(u.size());
        if ((rmse / u.mean()) < 1e-6) {
            std::cout << "Steady state reached at t = " << t << std::endl;
            shouldStop = true;
        }
    }
}