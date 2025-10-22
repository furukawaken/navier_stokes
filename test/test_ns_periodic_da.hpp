#include <algorithm>
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

// Forcing
// OUForcing2D.hpp  —  単一クラス版（Grid + OU Forcing 統合）
#pragma once
#include <Eigen/Dense>
#include <vector>
#include <random>
#include <cmath>
#include <stdexcept>
#include <algorithm>

inline Eigen::VectorXd tile_average_flat(
//inline Eigen::VectorXd tile_block_mean_replicate_by_size(
    const Eigen::VectorXd & field,
    int nx, int ny,
    int nnx=5, int nny=5
){
    if ( nx <= 0 || ny <= 0 )           throw std::invalid_argument( "nx, ny must be positive" );
    if ( nnx <= 0 || nny <= 0 )         throw std::invalid_argument( "nnx, nny must be positive" );
    if ( field.size() != nx * ny )      throw std::invalid_argument( "field size must be nx * ny" );

    Eigen::VectorXd out( nx * ny );

    // 小タイルの左上座標を nnx / nny 刻みで走査（端は余りサイズの小タイルになる）
    for ( int j0 = 0; j0 < ny; j0 += nny ) {
        int j1 = std::min( j0 + nny, ny );
        for ( int i0 = 0; i0 < nx; i0 += nnx ) {
            int i1 = std::min( i0 + nnx, nx );

            // 小タイル平均
            double sum = 0.0;
            for ( int j = j0; j < j1; ++j ) {
                int row = j * nx;
                for ( int i = i0; i < i1; ++i ) {
                    sum += field[ row + i ];
                }
            }
            int cnt = ( j1 - j0 ) * ( i1 - i0 );
            double mean = cnt > 0 ? sum / double( cnt ) : 0.0;

            // 小タイル領域を平均で埋める（元解像度に複製）
            for ( int j = j0; j < j1; ++j ) {
                int row = j * nx;
                for ( int i = i0; i < i1; ++i ) {
                    out[ row + i ] = mean;
                }
            }
        }
    }
    return out;
}

class OuForcing2D {
public:
    // コンストラクタ
    // nx, ny : 格子数（ベクトル長は nx*ny）
    // L      : 物理長（正方形 [0, L)^2）
    // kf, dk : 強制帯域 |k| ∈ [kf - dk, kf + dk]（整数格子上の半径）
    // m      : 使用モード本数（円環上の整数ベクトルからランダム抽出）
    // tau    : OU 相関時間（大きいほどゆっくり変化）
    // sigma  : OU 係数の定常標準偏差（外力強度のノブ）
    // seed   : 乱数シード（再現性確保）
    OuForcing2D( int nx, int ny, double L,
                 int kf, int dk, int m,
                 double tau = 1.0, double sigma = 0.03,
                 std::uint64_t seed = 42 )
    : nx_( nx ), ny_( ny ), L_( L ),
      dx_( L_ / nx_ ), dy_( L_ / ny_ ), kfac_( 2.0 * M_PI / L_ ),
      kf_( kf ), dk_( dk ), tau_( tau ), sigma_( sigma ),
      rng_( seed ), norm01_( 0.0, 1.0 ), uniPhi_( 0.0, 2.0 * M_PI )
    {
        if ( nx_ <= 0 || ny_ <= 0 )  throw std::invalid_argument( "nx, ny must be positive" );
        if ( L_  <= 0.0 )            throw std::invalid_argument( "L must be positive" );
        if ( kf_ <= 0 || dk_ < 0 )   throw std::invalid_argument( "kf > 0, dk >= 0 required" );
        if ( m    <= 0 )             throw std::invalid_argument( "m must be positive" );

        // 帯域 |k| ∈ [kf - dk, kf + dk] の整数波数を収集
        for ( int nx = -nx_ / 2; nx <= nx_ / 2; ++nx ) {
            for ( int ny = -ny_ / 2; ny <= ny_ / 2; ++ny ) {
                if ( nx == 0 && ny == 0 ) continue;
                double kmag = std::sqrt( double( nx * nx + ny * ny ) );
                if ( kmag >= kf_ - dk_ && kmag <= kf_ + dk_ ) {
                    shell_.emplace_back( nx, ny );
                }
            }
        }
        if ( shell_.empty() ) {
            throw std::runtime_error( "OuForcing2D: empty wavevector band — check kf, dk, nx, ny." );
        }

        // m 本のモードをランダム選択し、位相と OU 係数を初期化
        std::uniform_int_distribution<int> pick( 0, int( shell_.size() ) - 1 );
        modes_.reserve( m );
        for ( int i = 0; i < m; ++i ) {
            auto [inx, iny] = shell_[ pick( rng_ ) ];
            Mode md;
            md.nx  = inx;
            md.ny  = iny;
            md.kx  = kfac_ * double( inx );
            md.ky  = kfac_ * double( iny );
            md.phi = uniPhi_( rng_ );
            md.a   = 0.0;  // OU 係数初期値
            modes_.push_back( md );
        }
    }

    // === 1 行で：OU を 1 ステップ進めて外力を上書き生成 =====================
    void advanceAndFill( double dt, Eigen::VectorXd & f_u, Eigen::VectorXd & f_v ) {
        stepOu_( dt );
        this->computeForce_( f_u, f_v );
    }

    // === 分割して使いたい場合（多段積分など） ===============================
    void stepOu( double dt ) { stepOu_( dt ); }
    void compute( Eigen::VectorXd & f_u, Eigen::VectorXd & f_v ) const { this->computeForce_( f_u, f_v ); }

    // === チューニング用ユーティリティ =======================================
    void setSigma( double sigma ) { sigma_ = sigma; }
    void setTau( double tau )     { tau_   = tau;   }
    void reseed( std::uint64_t seed ) { rng_.seed( seed ); }
    void randomizePhases() { for ( auto & md : modes_ ) md.phi = uniPhi_( rng_ ); }
    void driftPhases( double dt, double rate = 0.1 ) { for ( auto & md : modes_ ) md.phi += rate * dt; }

    // デバッグ：div f の最大絶対値（中心差分・周期）
    double divergenceInfNorm( const Eigen::VectorXd & fX, const Eigen::VectorXd & fY ) const {
        auto at = [&]( int i, int j ) {
            if ( i < 0 ) i += nx_; if ( i >= nx_ ) i -= nx_;
            if ( j < 0 ) j += ny_; if ( j >= ny_ ) j -= ny_;
            return j * nx_ + i;
        };
        if ( fX.size() != nx_ * ny_ || fY.size() != nx_ * ny_ ) {
            throw std::runtime_error( "divergenceInfNorm: size mismatch" );
        }
        double maxAbs = 0.0;
        for ( int j = 0; j < ny_; ++j ) {
            for ( int i = 0; i < nx_; ++i ) {
                int ip = at( i + 1, j ), im = at( i - 1, j );
                int jp = at( i, j + 1 ), jm = at( i, j - 1 );
                double dfxdx = ( fX[ ip ] - fX[ im ] ) / ( 2.0 * dx_ );
                double dfydy = ( fY[ jp ] - fY[ jm ] ) / ( 2.0 * dy_ );
                maxAbs = std::max( maxAbs, std::abs( dfxdx + dfydy ) );
            }
        }
        return maxAbs;
    }

    // ゲッター
    int nx() const { return nx_; }
    int ny() const { return ny_; }
    double L() const { return L_; }
    double dx() const { return dx_; }
    double dy() const { return dy_; }

private:
    struct Mode {
        int nx, ny;     // 整数波数
        double kx, ky;  // 物理波数（2π / L 倍）
        double phi;     // 位相
        double a;       // OU 係数
    };

    void stepOu_( double dt ) {
        double rho = std::exp( -dt / tau_ );
        double sig = sigma_ * std::sqrt( std::max( 0.0, 1.0 - rho * rho ) );
        for ( auto & md : modes_ ) {
            md.a   = rho * md.a + sig * norm01_( rng_ );
            md.phi += 0.05 * dt * norm01_( rng_ );  // 微小ジッタ（対称ロック防止）
        }
    }

    // f = ∇^⊥ ψ_f を直接評価して上書き生成（div-free 保証：差分誤差のみ）
    void computeForce_( Eigen::VectorXd & f_u, Eigen::VectorXd & f_v ) const {
        int n = nx_ * ny_;
        f_u.setZero( n );
        f_v.setZero( n );

        for ( int j = 0; j < ny_; ++j ) {
            double y = j * dy_;
            for ( int i = 0; i < nx_; ++i ) {
                double x = i * dx_;
                double dpsidx = 0.0, dpsidy = 0.0;
                for ( const auto & md : modes_ ) {
                    double th = md.kx * x + md.ky * y + md.phi;
                    double s  = std::sin( th );
                    // ∂ψ/∂x = - Σ a kx sin(th),  ∂ψ/∂y = - Σ a ky sin(th)
                    dpsidx  += - md.a * md.kx * s;
                    dpsidy  += - md.a * md.ky * s;
                }
                int id = j * nx_ + i;
                f_u[ id ] =  dpsidy * 2.0;   // f_x =  ∂ψ/∂y
                f_v[ id ] = -dpsidx * 2.0;   // f_y = -∂ψ/∂x
            }
        }
    }

    // メンバ変数（末尾に _）
    int nx_, ny_;
    double L_, dx_, dy_, kfac_;
    int kf_, dk_;
    double tau_, sigma_;

    std::mt19937_64 rng_;
    std::normal_distribution<double> norm01_;
    std::uniform_real_distribution<double> uniPhi_;

    std::vector<std::pair<int,int>> shell_;
    std::vector<Mode> modes_;
};


// Compute convection term:
// div (uv cdot nabla u)
// = d / dx (u u) + d / dy (v u)
// div (uv cdot nabla v)
// = d / dx (u v) + d / dy (v v)
// xi  = x + dx/2
// eta = y + dy/2
void computeConvectionDA_(
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

void testNavierStokesPeriodicDA() {
    std::cout << "testNavierStokesPeriodicDA" << std::endl;

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
    const double reynolds  = 1000.0;
    const int    xmode     = 1;
    const int    ymode     = 1;
    const double lx        = nx * dx;
    const double ly        = ny * dy;
    const double kx        = 2.0 * M_PI * xmode / lx;
    const double ky        = 2.0 * M_PI * ymode / ly;
    const double k2        = kx * kx + ky * ky;
    double       tgf_scale = 1.0;

    // DA settings
    const double inflation = 10.0;

    // Turbulence parameters
    OuForcing2D forcing( nx, ny, dx * nx, /* kf = */ 16, /* dk = */ 1, /* m = */ 32,
                         /* tau = */ 1.0, /* sigma = */ 0.03, /* seed = */ 1234 );

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
    std::string filename{"../output/test8/ns.nc"};
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
    Eigen::VectorXd u_old   = Eigen::VectorXd::Zero(nx * ny);
    Eigen::VectorXd v_old   = Eigen::VectorXd::Zero(nx * ny);
    Eigen::VectorXd k1_u = Eigen::VectorXd::Zero(nx * ny);
    Eigen::VectorXd k2_u = Eigen::VectorXd::Zero(nx * ny);
    Eigen::VectorXd k3_u = Eigen::VectorXd::Zero(nx * ny);
    Eigen::VectorXd k4_u = Eigen::VectorXd::Zero(nx * ny);
    Eigen::VectorXd k1_v = Eigen::VectorXd::Zero(nx * ny);
    Eigen::VectorXd k2_v = Eigen::VectorXd::Zero(nx * ny);
    Eigen::VectorXd k3_v = Eigen::VectorXd::Zero(nx * ny);
    Eigen::VectorXd k4_v = Eigen::VectorXd::Zero(nx * ny);
    Eigen::VectorXd opt  = Eigen::VectorXd::Zero(nx * ny);
    Eigen::VectorXd opt2 = Eigen::VectorXd::Zero(nx * ny);
    Eigen::VectorXd opt3 = Eigen::VectorXd::Zero(nx * ny);
    Eigen::VectorXd opt4 = Eigen::VectorXd::Zero(nx * ny);

    // Setting initial condition
    Eigen::VectorXd u(tgf_u);
    Eigen::VectorXd v(tgf_v);
    Eigen::VectorXd u_da = Eigen::VectorXd::Zero(nx * ny);
    Eigen::VectorXd v_da = Eigen::VectorXd::Zero(nx * ny);
    p_manager.initialize(tgf_u, tgf_v);
    Eigen::VectorXd p(p_manager.getP());

    // Spin up
    std::cout << "Spinning up..." << std::endl;
    for (int it = 0; it < 10000; ++it) {
        /*----------------------SPIN UP----------------------*/
        forcing.advanceAndFill(dt, forcing_u, forcing_v);
        u_clone = u;
        v_clone = v;
        computeConvectionDA_(
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
        computeConvectionDA_(
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
    }
    
    // Saving initial condition
    writer.writeTime(t, writing_count);
    writer.writeU(std::vector<double>(u.data(), u.data() + u.size()), writing_count);
    writer.writeV(std::vector<double>(v.data(), v.data() + v.size()), writing_count);
    writer.writeP(std::vector<double>(p.data(), p.data() + p.size()), writing_count);
    opt  = u_da;
    opt2 = v_da;
    opt3 = tile_average_flat(u, nx, ny);
    opt4 = tile_average_flat(v, nx, ny);
    //opt2 = tile_average_flat(forcing_u, nx, ny);
    //opt = (u - tgf_u) / tgf_scale;
    writer.writeOptional(std::vector<double>( opt.data(),  opt.data()  + opt.size()),  writing_count);
    writer.writeOptional2(std::vector<double>(opt2.data(), opt2.data() + opt2.size()), writing_count);
    writer.writeOptional3(std::vector<double>(opt3.data(), opt3.data() + opt3.size()), writing_count);
    writer.writeOptional4(std::vector<double>(opt4.data(), opt4.data() + opt4.size()), writing_count);
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
        std::cout << "t = " << t << std::endl;
        for (int it = 0; it < si; ++it) {
            /*----------------------TRUE----------------------*/
            forcing.advanceAndFill(dt, forcing_u, forcing_v);
            u_clone = u;
            v_clone = v;
            computeConvectionDA_(
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
            computeConvectionDA_(
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

            /*----------------------DA----------------------*/
            u_old = u_clone;
            v_old = v_clone;
            u_clone = u_da;
            v_clone = v_da;
            computeConvectionDA_(
                nx, ny, dx, dy,
                u_da, v_da,
                nonlinear_term_u, nonlinear_term_v
            );
            total_forcing_u = - positive_laplacian * u_da
                              - nonlinear_term_u
                              + forcing_u
                              //+ tile_average_flat(forcing_u, nx, ny)
                              + inflation * tile_average_flat(
                                u_old - u_da,
                                nx,
                                ny
                              );
            total_forcing_v = - positive_laplacian * v_da
                              - nonlinear_term_v
                              + forcing_v
                              //+ tile_average_flat(forcing_v, nx, ny)
                              + inflation * tile_average_flat(
                                v_old - v_da,
                                nx,
                                ny
                              );
            p_manager.build(
                total_forcing_u,
                total_forcing_v
            );
            k1_u = dt * total_forcing_u;
            k1_v = dt * total_forcing_v;
            //
            u_tmp = u_clone + 0.5 * k1_u;
            v_tmp = v_clone + 0.5 * k1_v;
            computeConvectionDA_(
                nx, ny, dx, dy,
                u_tmp, v_tmp,
                nonlinear_term_u, nonlinear_term_v
            );
            total_forcing_u = - positive_laplacian * u_tmp
                              - nonlinear_term_u
                              + forcing_u
                              //+ tile_average_flat(forcing_u, nx, ny)
                              + inflation * tile_average_flat(
                                u_old - u_tmp,
                                nx,
                                ny
                              );
            total_forcing_v = - positive_laplacian * v_tmp
                              - nonlinear_term_v
                              + forcing_v
                              //+ tile_average_flat(forcing_v, nx, ny)
                              + inflation * tile_average_flat(
                                v_old - v_tmp,
                                nx,
                                ny
                              );
            p_manager.build(
                total_forcing_u,
                total_forcing_v
            );
            k2_u = dt * total_forcing_u;
            k2_v = dt * total_forcing_v;
            u_da = u_clone + k2_u;
            v_da = v_clone + k2_v;
            /*
            // k1
            computeConvectionDA_(
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
            computeConvectionDA_(
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
            computeConvectionDA_(
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
            computeConvectionDA_(
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
            //InitialVelocityAssembler::taylorGreenFlow(
            //    nx, ny, dx, dy, reynolds,
            //    1.0, 1.0, // Fourier modes
            //    tgf_u, tgf_v, tgf_p,
            //    t
            //);
            //tgf_scale = std::exp(- k2 * t / reynolds);
        }

        // Saving calculations
        writer.writeTime(t, writing_count);
        writer.writeU(std::vector<double>(u.data(), u.data() + u.size()), writing_count);
        writer.writeV(std::vector<double>(v.data(), v.data() + v.size()), writing_count);
        p_manager.initialize(u, v);
        p = p_manager.getP();
        writer.writeP(std::vector<double>(p.data(), p.data() + p.size()), writing_count);
        opt  = u_da;
        opt2 = v_da;
        opt3 = tile_average_flat(u_old, nx, ny);
        opt4 = tile_average_flat(v_old, nx, ny);
        //opt2 = forcing_v;
        //opt2 = tile_average_flat(forcing_u, nx, ny);
        //opt = (u - tgf_u) / tgf_scale;
        writer.writeOptional(std::vector<double>(opt.data(), opt.data() + opt.size()), writing_count);
        writer.writeOptional2(std::vector<double>(opt2.data(), opt2.data() + opt2.size()), writing_count);
        writer.writeOptional3(std::vector<double>(opt3.data(), opt3.data() + opt3.size()), writing_count);
        writer.writeOptional4(std::vector<double>(opt4.data(), opt4.data() + opt4.size()), writing_count);
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
