#include "PressureManager.hpp"

/*------------------------------------------------------------------------
--------------------------------SMAC METHOD-------------------------------
-------------------------------------------------------------------------*/

//Memo:
//using SpMat  = Eigen::SparseMatrix<double>;
//using Solver = Eigen::SimplicialLLT<SpMat>;
//using List   = std::vector<Eigen::Triplet<double>>;
PressureManagerSMAC::PressureManagerSMAC(
    const int nx, const int ny,
    const double dx, const double dy, const double dt,
    const List& positive_laplacian_triplet_list
) :
    nx_(nx), ny_(ny),
    dx_(dx), dy_(dy),
    dt_(dt),
    p_(std::make_unique<Eigen::VectorXd>(Eigen::VectorXd::Zero(nx * ny))),
    dpdx_(std::make_unique<Eigen::VectorXd>(Eigen::VectorXd::Zero(nx * ny))),
    dpdy_(std::make_unique<Eigen::VectorXd>(Eigen::VectorXd::Zero(nx * ny))),
    ddpdx_(std::make_unique<Eigen::VectorXd>(Eigen::VectorXd::Zero(nx * ny))),
    ddpdy_(std::make_unique<Eigen::VectorXd>(Eigen::VectorXd::Zero(nx * ny))),
    p_solver_(std::make_unique<Solver>())
{
    SpMat pinned_positive_laplacian(nx * ny, nx * ny);
    pinned_positive_laplacian.setFromTriplets(
        positive_laplacian_triplet_list.begin(),
        positive_laplacian_triplet_list.end()
    );
    BCModificationToolsNeumann::pinMatrix(pinned_positive_laplacian, nx, ny);
    pinned_positive_laplacian.makeCompressed();

    // Making a solver
    p_solver_->compute(pinned_positive_laplacian);
    if (p_solver_->info() != Eigen::Success) {
        throw std::runtime_error("SimplicialLLT factorization failed");
    }
}

void PressureManagerSMAC::setNabla(
    Eigen::VectorXd& dpdx,
    Eigen::VectorXd& dpdy,
    const Eigen::VectorXd& p

) {
    auto index = [this](int ix, int jy) {
        return ((ny_ + jy) % ny_) * nx_+ (nx_ + ix) % nx_;
    };

    const double dx_inv = 1.0 / dx_;
    const double dy_inv = 1.0 / dy_;
    #pragma omp parallel for schedule(static)
    for (int jy= 0; jy < ny_; jy++) {
        for (int ix= 0; ix < nx_; ix++) {
            dpdx[index(ix, jy)]
                = (
                      p[index(ix,     jy)] // (x, y)-coordinate
                    - p[index(ix - 1, jy)] // (x, y)-coordinate
                ) * dx_inv; // (xi, y)-coordinate
            dpdy[index(ix, jy)]
                = (
                      p[index(ix, jy    )] // (x, y)-coordinate
                    - p[index(ix, jy - 1)] // (x, y)-coordinate
                ) * dy_inv; // (x, eta)-coordinate
        }
    }
}

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
/* Projection method:
    dU_n/dt = Delta U_n + P f_n
*/
void PressureManagerSMAC::build(
    const Eigen::VectorXd& u,
    const Eigen::VectorXd& v
) {
    auto index = [this](int ix, int jy) {
        return ((ny_ + jy) % ny_) * nx_+ (nx_ + ix) % nx_;
    };

    // Settings of variables (each length = nx * ny)
    const double dx_inv = 1.0 / dx_;
    const double dy_inv = 1.0 / dy_;
    Eigen::VectorXd div_uv(nx_ * ny_);
    Eigen::VectorXd dp(nx_ * ny_);
    
    // Calculating div(u, v)
    // xi  = x - dx/2
    // eta = y - dy/2
    #pragma omp parallel for schedule(static)
    for (int jy= 0; jy < ny_; jy++) {
        for (int ix= 0; ix < nx_; ix++) {
            div_uv[index(ix, jy)]
                = (
                      u[index(ix + 1, jy)] // (xi, y)-coordinate
                    - u[index(ix,     jy)] // (xi, y)-coordinate
                ) * dx_inv // (x, y)-coordinate
                + (
                      v[index(ix, jy + 1)] // (x, eta)-coordinate
                    - v[index(ix, jy)]     // (x, eta)-coordinate
                ) * dy_inv; // (x, y)-coordinate
        }
    }
    div_uv[0] = 0.0; // Pinning
    
    // Updating dp and p    
    //auto t1 = std::chrono::high_resolution_clock::now();
    dp = p_solver_->solve(- (1.0 / dt_) * div_uv);
    *p_ += dp;
    (*p_).array() -= (*p_).mean(); // Making average free
    //auto t2 = std::chrono::high_resolution_clock::now();
    //std::chrono::duration<double> elapsed = t2 - t1;
    //std::cout
    //    << "Poisson solve took "
    //    << elapsed.count()
    //    << " seconds"
    //    << std::endl;

    // Calculating nabla p
    this->setNabla(*dpdx_, *dpdy_, *p_);

    // Calculating nabla dp
    this->setNabla(*ddpdx_, *ddpdy_, dp);
}

void PressureManagerSMAC::modifyVectorField(
    Eigen::VectorXd& u,
    Eigen::VectorXd& v
) const {
    u -= dt_ * (*ddpdx_);
    v -= dt_ * (*ddpdy_);
}

void PressureManagerSMAC::initialize(
    const Eigen::VectorXd& u,
    const Eigen::VectorXd& v
) {
    auto index = [this](int ix, int jy) {
        return ((ny_ + jy) % ny_) * nx_+ (nx_ + ix) % nx_;
    };

    // Settings of variables (each length = nx * ny)
    const double dx_inv = 1.0 / dx_;
    const double dy_inv = 1.0 / dy_;
    Eigen::VectorXd div_convection(nx_ * ny_);
    
    // Calculating div(uv cdot nabla uv
    // div(uv cdot nabla uv)
    // = (dudx)^2 + 2 * dudy * dvdx + (dvdy)^2
    #pragma omp parallel for schedule(static)
    for (int jy = 0;jy < ny_; ++jy){
        for (int ix = 0;ix < nx_; ++ix){
            const int c  = index(ix, jy);
            //
            double dudx = (
                  u[index(ix + 1, jy)]
                - u[index(ix, jy)]
            ) * dx_inv;
            //
            double dvdy = (
                  v[index(ix, jy + 1)]
                - v[index(ix, jy)]
            ) * dy_inv;
            //
            double u_up_x_y = (
                  u[index(ix + 1, jy + 1)]
                + u[index(ix,     jy + 1)]
            ) * 0.5;
            double u_low_x_y = (
                  u[index(ix + 1, jy - 1)]
                + u[index(ix,     jy -1)]
            ) * 0.5;
            double dudy = (
                  u_up_x_y
                - u_low_x_y 
            ) * (0.5 * dy_inv);
            //
            double v_right_x_y = (
                  v[index(ix + 1, jy + 1)]
                + v[index(ix + 1,     jy)]
            ) * 0.5;
            double v_left_x_y = (
                  v[index(ix - 1, jy + 1)]
                + v[index(ix - 1, jy)]
            ) * 0.5;
            double dvdx = (
                  v_right_x_y
                - v_left_x_y 
            ) * (0.5 * dx_inv);

            div_convection[c]
            = dudx * dudx
            + dudy * dvdx * 2.0
            + dvdy * dvdy;
        }
    }
    div_convection[0] = 0.0; // Pinning
    
    // Updating p    
    *p_ = p_solver_->solve(div_convection);
    (*p_).array() -= (*p_).mean(); // Making average free
}

/*-------------------------------------------------------------------------
-----------------------------PROJECTION METHOD-----------------------------
-------------------------------------------------------------------------*/
//Memo:
//using SpMat  = Eigen::SparseMatrix<double>;
//using Solver = Eigen::SimplicialLLT<SpMat>;
//using List   = std::vector<Eigen::Triplet<double>>;

/*
    dU/dt + nabla p = \Delta U - U cdot nabla U + f
              dU/dt = P (\Delta U - U cdot nabla U + f)
            nabla p = Q (\Delta U - U cdot nabla U + f)
        p = Q (\Delta U - U cdot nabla U + f)
          =    \Delta U - U cdot nabla U + f
          - P (\Delta U - U cdot nabla U + f)
    div U = 0

    P : Helmholtz projection
    P = I + \nabla (- Delta)^{-1} div
    Q = - \nabla (- Delta)^{-1} div
*/
PressureManagerProjection::PressureManagerProjection(
    const int nx, const int ny,
    const double dx, const double dy, const double dt,
    const List& positive_laplacian_triplet_list
) :
    nx_(nx), ny_(ny),
    dx_(dx), dy_(dy),
    dt_(dt),
    p_(std::make_unique<Eigen::VectorXd>(Eigen::VectorXd::Zero(nx * ny))),
    dpdx_(std::make_unique<Eigen::VectorXd>(Eigen::VectorXd::Zero(nx * ny))),
    dpdy_(std::make_unique<Eigen::VectorXd>(Eigen::VectorXd::Zero(nx * ny))),
    //ddpdx_(std::make_unique<Eigen::VectorXd>(Eigen::VectorXd::Zero(nx * ny))),
    //ddpdy_(std::make_unique<Eigen::VectorXd>(Eigen::VectorXd::Zero(nx * ny))),
    p_solver_(std::make_unique<Solver>())
{
    SpMat pinned_positive_laplacian(nx * ny, nx * ny);
    pinned_positive_laplacian.setFromTriplets(
        positive_laplacian_triplet_list.begin(),
        positive_laplacian_triplet_list.end()
    );
    BCModificationToolsNeumann::pinMatrix(pinned_positive_laplacian, nx, ny);
    pinned_positive_laplacian.makeCompressed();

    // Making a solver
    p_solver_->compute(pinned_positive_laplacian);
    if (p_solver_->info() != Eigen::Success) {
        throw std::runtime_error("SimplicialLLT factorization failed");
    }
}

void PressureManagerProjection::setNabla(
    Eigen::VectorXd& dpdx,
    Eigen::VectorXd& dpdy,
    const Eigen::VectorXd& p

) {
    auto index = [this](int ix, int jy) {
        return ((ny_ + jy) % ny_) * nx_+ (nx_ + ix) % nx_;
    };

    const double dx_inv = 1.0 / dx_;
    const double dy_inv = 1.0 / dy_;
    #pragma omp parallel for schedule(static)
    for (int jy= 0; jy < ny_; jy++) {
        for (int ix= 0; ix < nx_; ix++) {
            dpdx[index(ix, jy)]
                = (
                      p[index(ix,     jy)] // (x, y)-coordinate
                    - p[index(ix - 1, jy)] // (x, y)-coordinate
                ) * dx_inv; // (xi, y)-coordinate
            dpdy[index(ix, jy)]
                = (
                      p[index(ix, jy    )] // (x, y)-coordinate
                    - p[index(ix, jy - 1)] // (x, y)-coordinate
                ) * dy_inv; // (x, eta)-coordinate
        }
    }
}

void PressureManagerProjection::setDivergence(
    Eigen::VectorXd& div_uv,
    const Eigen::VectorXd& u,
    const Eigen::VectorXd& v
) {
    auto index = [this](int ix, int jy) {
        return ((ny_ + jy) % ny_) * nx_+ (nx_ + ix) % nx_;
    };

    const double dx_inv = 1.0 / dx_;
    const double dy_inv = 1.0 / dy_;
    #pragma omp parallel for schedule(static)
    for (int jy= 0; jy < ny_; jy++) {
        for (int ix= 0; ix < nx_; ix++) {
            div_uv[index(ix, jy)]
                = (
                      u[index(ix + 1, jy)] // (xi, y)-coordinate
                    - u[index(ix,     jy)] // (xi, y)-coordinate
                ) * dx_inv; // (x, y)-coordinate
                + (
                      v[index(ix, jy + 1)] // (x, eta)-coordinate
                    - v[index(ix, jy)]     // (x, eta)-coordinate
                ) * dy_inv; // (x, y)-coordinate
        }
    }
}

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
void PressureManagerProjection::build(
    Eigen::VectorXd& forcing_u, // \Delta u - (u,v) cdot nabla u + f_u
    Eigen::VectorXd& forcing_v  // \Delta v - (u,v) cdot nabla v + f_v
) {
    auto index = [this](int ix, int jy) {
        return ((ny_ + jy) % ny_) * nx_+ (nx_ + ix) % nx_;
    };

    // Settings of variables (each length = nx * ny)
    const double dx_inv = 1.0 / dx_;
    const double dy_inv = 1.0 / dy_;
    Eigen::VectorXd div_f(nx_ * ny_);
    
    // Calculating div(u, v)
    // xi  = x - dx/2
    // eta = y - dy/2
    #pragma omp parallel for schedule(static)
    //#pragma omp parallel for collapse(2) schedule(static)
    for (int jy= 0; jy < ny_; jy++) {
        for (int ix= 0; ix < nx_; ix++) {
            div_f[index(ix, jy)]
                = (
                      forcing_u[index(ix + 1, jy)] // (xi, y)-coordinate
                    - forcing_u[index(ix,     jy)] // (xi, y)-coordinate
                ) * dx_inv // (x, y)-coordinate
                + (
                      forcing_v[index(ix, jy + 1)] // (x, eta)-coordinate
                    - forcing_v[index(ix, jy)]     // (x, eta)-coordinate
                ) * dy_inv; // (x, y)-coordinate
        }
    }
    div_f[0] = 0.0; // Pinning
    
    // Updating dp and p    
    //auto t1 = std::chrono::high_resolution_clock::now();
    *p_ = - (p_solver_->solve(div_f));
    (*p_).array() -= (*p_).mean(); // Making average free
    /*
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = t2 - t1;
    std::cout
        << "Poisson solve took "
        << elapsed.count()
        << " seconds"
        << std::endl;
    */
    this->setNabla(*dpdx_, *dpdy_, *p_);
    forcing_u -= *dpdx_;
    forcing_v -= *dpdy_;
}

void PressureManagerProjection::initialize(
    const Eigen::VectorXd& u,
    const Eigen::VectorXd& v
) {
    auto index = [this](int ix, int jy) {
        return ((ny_ + jy) % ny_) * nx_+ (nx_ + ix) % nx_;
    };

    // Settings of variables (each length = nx * ny)
    const double dx_inv = 1.0 / dx_;
    const double dy_inv = 1.0 / dy_;
    Eigen::VectorXd div_convection(nx_ * ny_);
    
    // Calculating div(uv cdot nabla uv
    // div(uv cdot nabla uv)
    // = (dudx)^2 + 2 * dudy * dvdx + (dvdy)^2
    #pragma omp parallel for schedule(static)
    for (int jy = 0;jy < ny_; ++jy){
        for (int ix = 0;ix < nx_; ++ix){
            const int c  = index(ix, jy);
            //
            double dudx = (
                  u[index(ix + 1, jy)]
                - u[index(ix, jy)]
            ) * dx_inv;
            //
            double dvdy = (
                  v[index(ix, jy + 1)]
                - v[index(ix, jy)]
            ) * dy_inv;
            //
            double u_up_x_y = (
                  u[index(ix + 1, jy + 1)]
                + u[index(ix,     jy + 1)]
            ) * 0.5;
            double u_low_x_y = (
                  u[index(ix + 1, jy - 1)]
                + u[index(ix,     jy -1)]
            ) * 0.5;
            double dudy = (
                  u_up_x_y
                - u_low_x_y 
            ) * (0.5 * dy_inv);
            //
            double v_right_x_y = (
                  v[index(ix + 1, jy + 1)]
                + v[index(ix + 1,     jy)]
            ) * 0.5;
            double v_left_x_y = (
                  v[index(ix - 1, jy + 1)]
                + v[index(ix - 1, jy)]
            ) * 0.5;
            double dvdx = (
                  v_right_x_y
                - v_left_x_y 
            ) * (0.5 * dx_inv);

            div_convection[c]
            = dudx * dudx
            + dudy * dvdx * 2.0
            + dvdy * dvdy;
        }
    }
    div_convection[0] = 0.0; // Pinning
    
    // Updating p    
    *p_ = p_solver_->solve(div_convection);
    (*p_).array() -= (*p_).mean(); // Making average free
}