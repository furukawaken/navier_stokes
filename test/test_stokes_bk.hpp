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
#include "../src/Globals.hpp"

#include "../src/BoundaryConditionChecker.hpp"
#include "../src/BoundaryForcingBuilderCellCentered.hpp"
#include "../src/BoundaryForcingBuilderStaggered.hpp"
#include "../src/BoundaryOperations.hpp"
#include "../src/DataWriter.hpp"
#include "../src/DifferentialOperators.hpp"
#include "../src/IndexMapping.hpp"
#include "../src/IntegrationSchemes.hpp"
#include "../src/LinearSystemSolver.hpp"
#include "../src/MathFunctions.hpp"
#include "../src/ODEPropagator.hpp"
#include "../src/Parameters.hpp"
#include "../src/PressureManager.hpp"
#include "../src/SteadyForcingFactory.hpp"
#include "../src/TripletListFactory.hpp"
#include "../src/VectorSpaceFactory.hpp"

/* SMAC method:
    Navier-Stokes equations:
    (U_n+1 - U_n) / dt = Delta U_n - nabla p_n+1 + f_n
    div U_n+1 = 0
    --->
      (U_ast - U_n) / dt = Delta U_n - nabla p_n + f_n // O(dt^5)
    (U_n+1 - U_ast) / dt = - nabla (p_n+1 - p_n)
    (U_ast satisfy the same boundary condition as U_n+1)
    
    Taking div to the second eq: (dp := p_n+1 - p_n, nu: outer normal)
          - Delta dp = - div U_ast / dt, in interior   // O(dt^4),
    nabla dp cdot nu = 0,                on boundaries            
               p_n+1 = p_n + dp.                       // O(dt^4)
    
    (Remark)
    Since the Poisson eq is linear, we can put dp' := dt * dp, then
    - Delta dp' = - div U_ast     // O(dt^4)
         p_n+1  = p_n + dp' / dt. // O(dt^4)

    Summary:
         U_ast = U_n + dt * (Delta U_n - nabla p_n + f_n), // Solve ODE.
    - Delta dp = - div U_ast / dt,                         // Solve Poisson
         p_n+1 = p_n + dp,                                 // Update p
         U_n+1 = U_ast - dt * nabla dp,                    // Update U

    Arrays:
        u, v, p,
        u_new, v_new, p_new,
        <- u*, v*, dp
        div U = div (u, v)^T, 
        nabla_p <- (p_extended),

        top_u, bottom_u, left_u, right_u, <- given
        top_v, bottom_v, left_v, right_v, <- given
        top_p, bottom_p, left_p, right_p, <- from u, v
*/
template <typename VectorType>
void runStokes(
        NetCDFWriterStokes& writer,
        const Parameters& parameters,
        VectorSpaceFactory<std::vector<VectorType>, std::vector<size_t>>& vector_sp_factory_velocity,
        VectorSpaceFactory<VectorType, size_t>& vector_sp_factory_scalar_field,
        const std::string& filename
) {
    // Writing settings:
    int max_writing_count{3};
    int writing_count{0};
    int forwarding_step{1};

    // For time management: 
    int    it{0};
    double t{0.0};

    // Abbreviation
    const int    nx_u         = parameters.nx - 1;
    const int    ny_u         = parameters.ny;
    const int    nx_v         = parameters.nx;
    const int    ny_v         = parameters.ny - 1;
    const int    nx_p         = parameters.nx;
    const int    ny_p         = parameters.ny;
    const int    nt           = parameters.nt;
    const double dt           = parameters.dt;
    const double dx           = parameters.dx;
    const double dy           = parameters.dy;
    const double reynolds     = parameters.reynolds;
    const int    dimension    = 2;
    const int saving_interval = parameters.saving_interval;

    // Vector size(shape) for vector factories:
    size_t p_size{static_cast<size_t>(nx_p * ny_p)};
    std::vector<size_t> velocity_size{
        static_cast<size_t>(nx_u * ny_u),
        static_cast<size_t>(nx_v * ny_v)
    };
    size_t v_top_bottom_size{static_cast<size_t>(nx_v)};

    // Set vector lib:
    auto velocity_factory     = std::move(vector_sp_factory_velocity.createVectorFactory());
    auto velocity_algebra     = std::move(vector_sp_factory_velocity.createAlgebra());
    auto scalar_field_factory = std::move(vector_sp_factory_scalar_field.createVectorFactory());
    auto scalar_field_algebra = std::move(vector_sp_factory_scalar_field.createAlgebra());

    // (Initial) Vectors (Now zero to be edited):
    auto velocity_field = velocity_factory->createZeros(velocity_size);

    // Vectors for boundary values:
    auto top_val_v    = scalar_field_factory->createZeros(nx_v);
    auto bottom_val_v = scalar_field_factory->createZeros(nx_v);
    auto left_val_u   = scalar_field_factory->createZeros(ny_u);
    auto right_val_u  = scalar_field_factory->createZeros(ny_u);

    //
    std::vector<VectorType> interior_forcings{
        scalar_field_factory->createOnes(nx_u * ny_u),
        scalar_field_factory->createZeros(nx_v * ny_v)
    };

    // Set evolution solvers:
    StokesPropagator velocity_propagator(
        std::vector<LinearHeatPropagator>{
            LinearHeatPropagator(
                nx_u,
                ny_u,
                HorizontalVelocityTripletFactory().getDirichletList(
                    nx_u,
                    ny_u,
                    dx,
                    dy,
                    reynolds
                ),
                scalar_field_factory->createZeros(nx_u * ny_u),
                BoundaryForcingBuilderNCxCCy(
                    nx_u, ny_u,
                    dx,   dy,
                    reynolds
                ).getDirichletBoundaryForcingFrom(
                    scalar_field_factory->createZeros(nx_u),
                    scalar_field_factory->createZeros(nx_u),
                    scalar_field_factory->createZeros(ny_u),
                    scalar_field_factory->createZeros(ny_u)
                )
            ),
            LinearHeatPropagator(
                nx_v,
                ny_v,
                VerticalVelocityTripletFactory().getDirichletList(
                    nx_v,
                    ny_v,
                    dx,
                    dy,
                    reynolds
                ),
                scalar_field_factory->createZeros(nx_v * ny_v),
                BoundaryForcingBuilderCCxNCy(
                    nx_v, ny_v,
                    dx,   dy,
                    parameters.reynolds
                ).getDirichletBoundaryForcingFrom(
                    scalar_field_factory->createZeros(nx_v),
                    scalar_field_factory->createZeros(nx_v),
                    scalar_field_factory->createZeros(ny_v),
                    scalar_field_factory->createZeros(ny_v)
                )
            )
        }
    );
    PressureManager p_manager(
        nx_p, ny_p,
        dx, dy, dt
    );

    // Update
    p_manager.build(
        velocity_field,
        top_val_v,  bottom_val_v,
        left_val_u, right_val_u
    );
    velocity_field = p_manager.getModifedVectorField(velocity_field);

    // Save t, u, v, p:
    writer.writeTime(t, writing_count);
    writer.writeVectorField(
        velocity_factory->convertToSTDVectorList(velocity_field),
        writing_count
    );
    writer.writeP(
        scalar_field_factory->convertToSTDVectorList(p_manager.getP())[0],
        writing_count
    );
    writer.writeData(
        scalar_field_factory->convertToSTDVectorList(
            p_manager.getDiv()
        )[0],
        writing_count,
        "div"
    ); // debug
    writing_count++;

    // Routine
    while (it < nt)
    {
        std::cout << it << std::endl;
        // Integration
        IntegrationSchemes::updateByRungeKutta4ForStokes<std::vector<VectorType>, StokesPropagator, PressureManager>(
            velocity_field,
            velocity_propagator,
            p_manager,
            t,
            dt,
            forwarding_step, // forwarding step
            interior_forcings,
            vector_sp_factory_velocity
        );
        it += forwarding_step;
        
        // Save t, u, v, p:
        if (it % saving_interval == 0) {
            writer.writeTime(t, writing_count);
            writer.writeVectorField(
                velocity_factory->convertToSTDVectorList(velocity_field),
                writing_count
            );
            writer.writeP(
                scalar_field_factory->convertToSTDVectorList(p_manager.getP())[0],
                writing_count
            );
            writer.writeData(
                scalar_field_factory->convertToSTDVectorList(
                    p_manager.getDiv()
                )[0],
                writing_count,
                "div"
            ); // debug
            writing_count++;
        }
        if (max_writing_count < writing_count) {
            break;
        }
    }
}

void testStokes()
{
    std::cout << "testStokesMixedDirichletNeumann" << std::endl;

    // Parameters settings
    Parameters parameters;
    parameters.nt = 100000;
    parameters.nx = 50;
    parameters.ny = 50;
    parameters.dx = 0.1;
    parameters.dy = 0.1;
    parameters.dt = 0.02;
    parameters.reynolds = 10.0;
    parameters.saving_interval = 1;
    parameters.checkStokesCFL();

    // Make ncfile:
    std::string filename{"../output/test6/stokes_dn.nc"};
    NetCDFWriterStokes writer(
        filename,
        parameters.dt, parameters.dy, parameters.dx,
                       parameters.ny, parameters.nx
    );
    writer.createFile();
    writer.setCoordinate();

    // Set vector lib
    EigenVectorSpaceListFactory vector_sp_factory_velocity;
    EigenVectorSpaceFactory     vector_sp_factory_pressure;

    runStokes<Eigen::VectorXd>(writer, parameters, vector_sp_factory_velocity, vector_sp_factory_pressure, filename);
}