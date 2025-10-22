#include <cmath>
#include <cstdio>
#include <fstream>
#include <execution>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>
#include <iostream>
#include <memory>
#include <netcdf.h>
#include <stdexcept>
#include <vector>
#include <utility>
#include <omp.h>

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


void testLinearHeatMixedDirichletNeumann() {
    std::cout << "testLinearHeatMixedDirichletNeumann" << std::endl;

    // Saving settings
    int max_writing_count{100};
    int writing_count{0};
    double t{0.0};
    bool shouldStop = false;

    // Parameters settings
    Parameters parameters;
    int    si   = 100;
    int    nt   = max_writing_count * si;
    int    nx   = 100;
    int    ny   = 100;
    double dt   = 0.001;
    double dx   = 0.1;
    double dy   = 0.1;
    double nu = 1.0;

    // Inserting parameters
    parameters.nt              = nt;
    parameters.nx              = nx;
    parameters.ny              = ny;
    parameters.dt              = dt;
    parameters.dx              = dx;
    parameters.dy              = dy;
    parameters.diffusivity     = nu;
    parameters.saving_interval = si;

    // Setting a boundary condition
    BoundaryConditionChecker bc_checker(
        std::unordered_map<std::string, std::string>{
            {"t", "D"},
            {"b", "D"},
            {"l", "D"},
            {"r", "D"}
        }
    );

    // Making ncfile:
    std::string filename{"../output/test3/heat_dn.nc"};
    NetCDFWriter writer(
        filename,
        parameters.dt,
        parameters.dy, parameters.dx,
        parameters.ny, parameters.nx
    );
    writer.createFile(); 
    writer.setCoordinate();

    // Setting solution vectors
    Eigen::VectorXd u
        = InitialFunctionFactory::cosHill(nx, ny, dx, dy); // Initial data
        //= InitialFunctionFactory::twinCosHill(nx, ny, dx, dy); // Initial data
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

    // Making a linear solver
    std::vector<Eigen::Triplet<double>> triplet_list =
        LaplacianTripletFactory().getList(
            nx, ny, dx, dy, nu, bc_checker
        ); // Positive Laplacian
    Eigen::SparseMatrix<double> positive_laplacian(nx * ny, nx * ny);
    positive_laplacian.setFromTriplets(triplet_list.begin(), triplet_list.end());
    positive_laplacian.makeCompressed();

    // Saving initial condition
    writer.writeTime(t, writing_count);
    writer.writeData(
        std::vector<double>(u.data(), u.data() + u.size()),
        writing_count
    );
    writing_count++;

    // Time integration by RK4
    for ([[maybe_unused]] int i = 0; i < max_writing_count && !shouldStop; ++i) {
        std::cout << "Integration count : " << i << std::endl;
        for (int it = 0; it < si; ++it) {
            u_clone = u;

            // Calculations of k1:
            k1 = - positive_laplacian * u
                 + total_forcing;

            // Calculations of k2:
            u  = u_clone + 0.5 * dt * k1;
            k2 = - positive_laplacian * u
                 + total_forcing;

            // Calculations of k3:
            u  = u_clone + 0.5 * dt * k2;
            k3 = - positive_laplacian * u
                 + total_forcing;

            // Calculations of k4:
            u  = u_clone + dt * k3;
            k4 = - positive_laplacian * u
                 + total_forcing;

            // Update u, t:
            u = u_clone + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
            t += dt;

            //if (u.maxCoeff() > 100) {
            //    std::cout << "Solution is blowing up at t = " << t << std::endl;
            //    std::cout << "u.maxCoeff() = " << u.maxCoeff() << std::endl;
            //    shouldStop = true;
            //    break; // break inner loop
            //}
        }

        // Saving calculations
        writer.writeTime(t, writing_count);
        writer.writeData(
            std::vector<double>(u.data(), u.data() + u.size()),
            writing_count
        );
        writing_count++;
    }
    /*
    // Parameters settings
    Parameters parameters;
    parameters.nt = 1001;
    parameters.dt = 0.01;
    parameters.nx = 50;
    parameters.ny = 50;
    parameters.dx = 0.1;
    parameters.dy = 0.1;
    parameters.forcing_amplitude = 1.0;
    parameters.saving_interval = 10;

    // Make ncfile:
    std::string filename{"../output/test5/heat_dn.nc"};
    NetCDFWriter writer(
        filename,
        parameters.dt,
        parameters.dy, parameters.dx,
        parameters.ny, parameters.nx
    );
    writer.createFile(); 
    writer.setCoordinate();

    // Set vector lib
    EigenVectorSpaceFactory vector_sp_factory;
    auto vector_factory = vector_sp_factory.createVectorFactory();

    // Boundary condition checker
    BoundaryConditionChecker bc_checker(
        std::unordered_map<std::string, std::string>{
            {"t", "D"},
            {"b", "N"},
            {"l", "D"},
            {"r", "N"}
        }
    );   

    // Set propagator
    LinearHeatPropagator heat_propagator(
        parameters.nx,
        parameters.ny,
        InhomogeneousHeatTripletFactory().getEvolutionMatrixList(
            parameters.nx, parameters.ny,
            parameters.dx, parameters.dy,
            parameters.diffusivity  , parameters.diffusivity  ,
            bc_checker
        ),
        SteadyForcingFactory().constantFunction(
            parameters.nx,
            parameters.ny,
            parameters.forcing_amplitude
        ),
        BoundaryForcingBuilderCellCentered(
            parameters.nx,  parameters.ny,
            parameters.dx,  parameters.dy,
            parameters.diffusivity  , parameters.diffusivity  ,
            bc_checker
        ).getBoundaryForcingFrom(
            vector_factory->createZeros(parameters.nx),
            vector_factory->createZeros(parameters.nx),
            vector_factory->createZeros(parameters.ny),
            vector_factory->createZeros(parameters.ny)
        )
    );

    runLinearInhomogeneousHeat<Eigen::VectorXd, LinearHeatPropagator>(writer, parameters, vector_sp_factory, heat_propagator, filename);
    */
}