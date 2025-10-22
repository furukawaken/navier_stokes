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
#include "../src/DataWriter.hpp"
#include "../src/InitialVelocityFactory.hpp"
#include "../src/Parameters.hpp"
#include "../src/SteadyForcingFactory.hpp"
#include "../src/TripletListFactory.hpp"
#include "../src/PressureManager.hpp"

// Compute convection term:
// div (uv cdot nabla u)
// = d / dx (u u) + d / dy (v u)
// div (uv cdot nabla v)
// = d / dx (u v) + d / dy (v v)
// xi  = x + dx/2
// eta = y + dy/2
void computeConvectionRK4_(
    const int    nx, const int    ny,
    const double dx, const double dy,
    const Eigen::VectorXd& u,
    const Eigen::VectorXd& v,
    Eigen::VectorXd& nonlinear_u,
    Eigen::VectorXd& nonlinear_v
) {
    const double dx_inverse = 1.0 / dx;
    const double dy_inverse = 1.0 / dy;
    nonlinear_u.setZero(nx*ny);
    nonlinear_v.setZero(nx*ny);

    auto index = [nx, ny](int ix, int jy) {
        return ((ny + jy) % ny) * nx+ (nx + ix) % nx;
    };

    #pragma omp parallel for schedule(static)
    for (int jy = 0;jy < ny; ++jy){
        for (int ix = 0;ix < nx; ++ix){
            const int c  = index(ix, jy);

            // Calculating nonlinear_u
            double u_right_for_u = ( u[index(ix + 1, jy)] + u[c] ) * 0.5; // (x, y)
            double u_left_for_u  = ( u[index(ix - 1, jy)] + u[c] ) * 0.5; // (x-1, y)
            double duudx = (
                  u_right_for_u * u_right_for_u
                - u_left_for_u  * u_left_for_u
            ) * dx_inverse; // (xi, y)

            double v_up_for_u  = ( v[index(ix - 1, jy + 1)] + v[index(ix, jy + 1)] ) * 0.5; // (xi, eta+1)
            double u_up_for_u  = ( u[index(ix, jy + 1)] + u[c] ) * 0.5;                     // (xi, eta+1)
            double v_low_for_u = ( v[index(ix - 1, jy)] + v[index(ix, jy)] ) * 0.5;         // (xi, eta)
            double u_low_for_u = ( u[index(ix, jy - 1)] + u[c] ) * 0.5;                     // (xi, eta)
            double dvudy = (
                  v_up_for_u  * u_up_for_u 
                - v_low_for_u * u_low_for_u
            ) * dy_inverse; // (xi, y)

            nonlinear_u[c] = duudx + dvudy;

            // Calculating nonlinear_v
            double u_right_for_v = ( u[index(ix + 1, jy)] + u[index(ix + 1, jy - 1)] ) * 0.5; // (xi+1, eta)
            double v_right_for_v = ( v[index(ix + 1, jy)] + v[c] ) * 0.5;                     // (xi+1, eta)
            double u_left_for_v  = ( u[c] + u[index(ix, jy - 1)] ) * 0.5;                     // (xi, eta)
            double v_left_for_v  = ( v[index(ix - 1, jy)] + v[c] ) * 0.5;                     // (xi, eta)
            double duvdx = (
                  u_right_for_v * v_right_for_v // (xi+1, eta)
                - u_left_for_v  * v_left_for_v  // (xi, eta) 
            ) * dx_inverse; // (x, eta)

            double v_up_for_v  = ( v[index(ix, jy + 1)] + v[c] ) * 0.5; // (x, y)
            double v_low_for_v = ( v[index(ix, jy - 1)] + v[c] ) * 0.5; // (x, y-1)
            double dvvdy = (
                  v_up_for_v  * v_up_for_v
                - v_low_for_v * v_low_for_v
            ) * dy_inverse; // (x, eta)

            nonlinear_v[c] = duvdx + dvvdy;
        }
    }
}

/*
void computeConvectionRK4_(
    const int    nx, const int    ny,
    const double dx, const double dy,
    const Eigen::VectorXd& u,
    const Eigen::VectorXd& v,
    Eigen::VectorXd& nonlinear_u,
    Eigen::VectorXd& nonlinear_v
) {
    const double half_dx_inverse = 0.5 / dx;
    const double half_dy_inverse = 0.5 / dy;
    nonlinear_u.setZero(nx * ny);
    nonlinear_v.setZero(nx * ny);

    auto index = [nx, ny](int ix, int jy) {
        return ((ny + jy) % ny) * nx + (nx + ix) % nx;
    };

    #pragma omp parallel for schedule(static)
    for (int jy = 0; jy < ny; ++jy){
        for (int ix = 0; ix < nx; ++ix){
            const int c  = index(ix, jy);

            // Calculating nonlinear_u
            // u dudx
            double dudx = (
                  u[index(ix + 1, jy)]
                - u[index(ix - 1, jy)]
            ) * half_dx_inverse;
            double ududx = u[c] * dudx;
            // v dudy
            double v_for_u  = (
                  v[index(ix,     jy    )]
                + v[index(ix,     jy + 1)]
                + v[index(ix - 1, jy    )]
                + v[index(ix - 1, jy + 1)]
            ) * 0.25;
            double dudy = (
                  u[index(ix, jy + 1)]
                - u[index(ix, jy - 1)]
            ) * half_dy_inverse;
            double vdudy = v_for_u * dudy;
            // u dudx + v dudy
            nonlinear_u[c] = ududx + vdudy;

            // Calculating nonlinear_v
            // u dvdx
            double u_for_v = (
                  u[index(ix,     jy    )]
                + u[index(ix + 1, jy    )]
                + u[index(ix,     jy - 1)]
                + u[index(ix + 1, jy - 1)]
            ) * 0.25;
            double dvdx = (
                  v[index(ix + 1, jy)]
                - v[index(ix - 1, jy)]
            ) * half_dx_inverse;
            double udvdx = u_for_v * dvdx;
            // v dvdy
            double dvdy = (
                  v[index(ix, jy + 1)]
                - v[index(ix, jy - 1)]
            ) * half_dy_inverse;
            double vdvdy = v[c] * dvdy;
            // udvdx + vdvdy
            nonlinear_v[c] = udvdx + vdvdy;
        }
    }
}
*/

void testNavierStokesPeriodicRK4() {
    std::cout << "testNavierStokesPeriodicRK4" << std::endl;

    // Saving settings
    int max_writing_count{100};
    int writing_count{0};
    double t{0.0};
    bool shouldStop = false;

    // Parameters settings
    Parameters parameters;
    const int    si        = 100;
    const int    nt        = max_writing_count * si;
    const int    nx        = 100;
    const int    ny        = 100;
    const double dt        = 0.001;
    const double dx        = M_PI * 0.01;
    const double dy        = M_PI * 0.01;
    const double reynolds  = 100.0;
    const int    xmode     = 1;
    const int    ymode     = 1;
    const double lx        = nx * dx;
    const double ly        = ny * dy;
    const double kx        = 2.0 * M_PI * xmode / lx;
    const double ky        = 2.0 * M_PI * ymode / ly;
    const double k2        = kx * kx + ky * ky;
    double       tgf_scale = 1.0;

    // Inserting parameters
    parameters.nt              = nt;
    parameters.nx              = nx;
    parameters.ny              = ny;
    parameters.dt              = dt;
    parameters.dx              = dx;
    parameters.dy              = dy;
    parameters.reynolds        = reynolds;
    parameters.saving_interval = si;

    // Setting a boundary condition
    BoundaryConditionChecker bc_checker("T");

    // Setting a writer
    std::string filename{"../output/test7/ns.nc"};
    NetCDFWriterNavierStokesPeriodic writer(
        filename,
        dt, dy, dx, ny, nx, reynolds
    );
    writer.createFile();
    writer.setCoordinate();

    // Settings for evolution operators
    Eigen::SparseMatrix<double> positive_laplacian(nx * ny, nx * ny);
    std::vector<Eigen::Triplet<double>> laplacian_list
        = LaplacianTripletFactory().getList(
            nx, ny, dx, dy, 1.0 / reynolds, bc_checker
        ); // Positive Laplacian :- Delta / Re
    positive_laplacian.setFromTriplets(laplacian_list.begin(), laplacian_list.end());
    positive_laplacian.makeCompressed();
    PressureManagerProjection p_manager(
        nx, ny,
        dx, dy, dt,
        //laplacian_list
        LaplacianTripletFactory().getList(
            nx, ny, dx, dy, 1.0, bc_checker
        ) // Positive Laplacian :- Delta
    );

    // Setting vectors
    Eigen::VectorXd tgf_u = Eigen::VectorXd::Zero(nx * ny);
    Eigen::VectorXd tgf_v = Eigen::VectorXd::Zero(nx * ny);
    Eigen::VectorXd tgf_p = Eigen::VectorXd::Zero(nx * ny);
    InitialVelocityAssembler::taylorGreenFlow(
        nx, ny, dx, dy, reynolds,
        1.0, 1.0, // Fourier modes
        tgf_u, tgf_v, tgf_p
    );
    Eigen::VectorXd forcing_u = Eigen::VectorXd::Zero(nx * ny);
    Eigen::VectorXd forcing_v = Eigen::VectorXd::Zero(nx * ny);
    Eigen::VectorXd nonlinear_term_u = Eigen::VectorXd::Zero(nx * ny);
    Eigen::VectorXd nonlinear_term_v = Eigen::VectorXd::Zero(nx * ny);
    Eigen::VectorXd total_forcing_u = Eigen::VectorXd::Zero(nx * ny);
    Eigen::VectorXd total_forcing_v = Eigen::VectorXd::Zero(nx * ny);
    Eigen::VectorXd u_clone = Eigen::VectorXd::Zero(nx * ny);
    Eigen::VectorXd v_clone = Eigen::VectorXd::Zero(nx * ny);
    Eigen::VectorXd u_tmp   = Eigen::VectorXd::Zero(nx * ny);
    Eigen::VectorXd v_tmp   = Eigen::VectorXd::Zero(nx * ny);
    Eigen::VectorXd k1_u = Eigen::VectorXd::Zero(nx * ny);
    Eigen::VectorXd k2_u = Eigen::VectorXd::Zero(nx * ny);
    Eigen::VectorXd k3_u = Eigen::VectorXd::Zero(nx * ny);
    Eigen::VectorXd k4_u = Eigen::VectorXd::Zero(nx * ny);
    Eigen::VectorXd k1_v = Eigen::VectorXd::Zero(nx * ny);
    Eigen::VectorXd k2_v = Eigen::VectorXd::Zero(nx * ny);
    Eigen::VectorXd k3_v = Eigen::VectorXd::Zero(nx * ny);
    Eigen::VectorXd k4_v = Eigen::VectorXd::Zero(nx * ny);
    Eigen::VectorXd opt  = Eigen::VectorXd::Zero(nx * ny);

    // Setting initial condition
    Eigen::VectorXd u(tgf_u);
    Eigen::VectorXd v(tgf_v);
    p_manager.initialize(tgf_u, tgf_v);
    Eigen::VectorXd p(p_manager.getP());
    
    // Saving initial condition
    writer.writeTime(t, writing_count);
    writer.writeU(std::vector<double>(u.data(), u.data() + u.size()), writing_count);
    writer.writeV(std::vector<double>(v.data(), v.data() + v.size()), writing_count);
    writer.writeP(std::vector<double>(p.data(), p.data() + p.size()), writing_count);
    opt = (u - tgf_u) / tgf_scale;
    writer.writeOptional(std::vector<double>(opt.data(), opt.data() + opt.size()), writing_count);
    writing_count++;

    // Time integration by RK4 by projection method
    /*
          dU/dt = \Delta U - U cdot nabla U - \nabla p + f
                = P (\Delta U - U cdot nabla U + f)
                + Q (\Delta U - U cdot nabla U + f)
                = P (\Delta U - U cdot nabla U + f)
                + nabla p
        nabla p = Q (\Delta U - U cdot nabla U + f)
          div U = 0

        Q = - \nabla (- Delta)^{-1} div
        P : Helmholtz projection
        P = I + \nabla (- Delta)^{-1} div
          = I - Q
    */
    for ([[maybe_unused]] int i = 0; i < max_writing_count && !shouldStop; ++i) {
        std::cout << "Integration count : " << t << std::endl;
        for (int it = 0; it < si; ++it) {
            u_clone = u;
            v_clone = v;

            computeConvectionRK4_(
                nx, ny, dx, dy,
                u, v,
                nonlinear_term_u, nonlinear_term_v
            );
            total_forcing_u = - positive_laplacian * u
                              - nonlinear_term_u
                              + forcing_u;
            total_forcing_v = - positive_laplacian * v
                              - nonlinear_term_v
                              + forcing_v;
            p_manager.build(
                total_forcing_u,
                total_forcing_v
            );
            k1_u = dt * total_forcing_u;
            k1_v = dt * total_forcing_v;
            //
            u_tmp = u_clone + 0.5 * k1_u;
            v_tmp = v_clone + 0.5 * k1_v;
            computeConvectionRK4_(
                nx, ny, dx, dy,
                u_tmp, v_tmp,
                nonlinear_term_u, nonlinear_term_v
            );
            total_forcing_u = - positive_laplacian * u_tmp
                              - nonlinear_term_u
                              + forcing_u;
            total_forcing_v = - positive_laplacian * v_tmp
                              - nonlinear_term_v
                              + forcing_v;
            p_manager.build(
                total_forcing_u,
                total_forcing_v
            );
            k2_u = dt * total_forcing_u;
            k2_v = dt * total_forcing_v;
            u = u_clone + k2_u;
            v = v_clone + k2_v;

            /*
            // k1
            computeConvectionRK4_(
                nx, ny, dx, dy,
                u, v,
                nonlinear_term_u, nonlinear_term_v
            );
            total_forcing_u = - positive_laplacian * u
                              - nonlinear_term_u
                              + forcing_u;
            total_forcing_v = - positive_laplacian * v
                              - nonlinear_term_v
                              + forcing_v;
            p_manager.build(
                total_forcing_u,
                total_forcing_v
            );
            k1_u = dt * total_forcing_u;
            k1_v = dt * total_forcing_v;
            // k2
            u_tmp.noalias() = u_clone + 0.5 * k1_u;
            v_tmp.noalias() = v_clone + 0.5 * k1_v;
            computeConvectionRK4_(
                nx, ny, dx, dy,
                u_tmp,
                v_tmp,
                nonlinear_term_u, nonlinear_term_v
            );
            total_forcing_u = - positive_laplacian * u_tmp
                              - nonlinear_term_u
                              + forcing_u;
            total_forcing_v = - positive_laplacian * v_tmp
                              - nonlinear_term_v
                              + forcing_v;
            p_manager.build(
                total_forcing_u,
                total_forcing_v
            );
            k2_u = dt * total_forcing_u;
            k2_v = dt * total_forcing_v;
            // k3
            u_tmp.noalias() = u_clone + 0.5 * k2_u;
            v_tmp.noalias() = v_clone + 0.5 * k2_v;
            computeConvectionRK4_(
                nx, ny, dx, dy,
                u_tmp,
                v_tmp,
                nonlinear_term_u, nonlinear_term_v
            );
            total_forcing_u = - positive_laplacian * u_tmp
                              - nonlinear_term_u
                              + forcing_u;
            total_forcing_v = - positive_laplacian * v_tmp
                              - nonlinear_term_v
                              + forcing_v;
            p_manager.build(
                total_forcing_u,
                total_forcing_v
            );
            k3_u = dt * total_forcing_u;
            k3_v = dt * total_forcing_v;
            // k4
            u_tmp.noalias() = u_clone + k3_u;
            v_tmp.noalias() = v_clone + k3_v;
            computeConvectionRK4_(
                nx, ny, dx, dy,
                u_tmp,
                v_tmp,
                nonlinear_term_u, nonlinear_term_v
            );
            total_forcing_u = - positive_laplacian * u_tmp
                              - nonlinear_term_u
                              + forcing_u;
            total_forcing_v = - positive_laplacian * v_tmp
                              - nonlinear_term_v
                              + forcing_v;
            p_manager.build(
                total_forcing_u,
                total_forcing_v
            );
            k4_u = dt * total_forcing_u;
            k4_v = dt * total_forcing_v;
            // Updating u, v
            u = u_clone + (k1_u + 2.0 * k2_u + 2.0 * k3_u + k4_u) / 6.0;
            v = v_clone + (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v) / 6.0;
            */
            t += dt;
            // Debug
            InitialVelocityAssembler::taylorGreenFlow(
                nx, ny, dx, dy, reynolds,
                1.0, 1.0, // Fourier modes
                tgf_u, tgf_v, tgf_p,
                t
            );
            tgf_scale = std::exp(- k2 * t / reynolds);
        }

        // Saving calculations
        writer.writeTime(t, writing_count);
        writer.writeU(std::vector<double>(u.data(), u.data() + u.size()), writing_count);
        writer.writeV(std::vector<double>(v.data(), v.data() + v.size()), writing_count);
        p_manager.initialize(u, v);
        p = p_manager.getP();
        writer.writeP(std::vector<double>(p.data(), p.data() + p.size()), writing_count);
        opt = (u - tgf_u) / tgf_scale;
        writer.writeOptional(std::vector<double>(opt.data(), opt.data() + opt.size()), writing_count);
        writing_count++;
    }
}
    /*
    // Time integration by RK4
    for ([[maybe_unused]] int i = 0; i < max_writing_count && !shouldStop; ++i) {
        std::cout << "Integration count : " << i << std::endl;
        for (int it = 0; it < si; ++it) {
            //shouldStop = true; break;
            u_clone = u;

            // Calculations of k1:
            k1 = - positive_laplacian * u
               + computeConvectionSMAC_(u, v, nx, ny, dx, dy);

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
               + computeConvectionSMAC_(u, v, nx, ny, dx, dy);

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
               + computeConvectionSMAC_(u, v, nx, ny, dx, dy);

            // Calculations of k4:
            u  = u_clone + dt * k3;
            v  = solver.solve(
                BCModificationToolsNeumann::getPinedVector(
                    u
                    - u.mean() * Eigen::VectorXd::Ones(nx * ny)
                )
            );
            k4 = - positive_laplacian * u
               + computeConvectionSMAC_(u, v, nx, ny, dx, dy);

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
*/
