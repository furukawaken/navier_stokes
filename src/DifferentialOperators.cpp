#include "DifferentialOperators.hpp"

std::vector<Eigen::VectorXd> DifferentialOperators::nablaOfScalerField(
    const Eigen::VectorXd& p,
    const int    nx,
    const int    ny,
    const double dx,
    const double dy
) {
    const int nx_nabla{nx - 1};
    const int ny_nabla{ny - 1};
    Eigen::VectorXd dpdx = Eigen::VectorXd::Zero(nx_nabla * ny      );
    Eigen::VectorXd dpdy = Eigen::VectorXd::Zero(nx       * ny_nabla);
    std::vector<Eigen::VectorXd> result{};
        
    for (int ip = 0; ip < (nx * ny); ++ip) {
        int ipx = ip % nx;
        int ipy = ip / nx;
        int i_nabla_x = ipx + nx_nabla * ipy;
        int i_nabla_y = ipx + nx       * ipy;
        
        if(((ipx + 1) % nx) != 0) {
            dpdx[i_nabla_x]
            = (
                  p[ip + 1]
                - p[ip]
            ) / dx;
        }

        if (ip < nx * (ny - 1)) {
            dpdy[i_nabla_y]
            = (
                  p[ip + nx]
                - p[ip]
            ) / dy;
        }
    }

    result.push_back(dpdx);
    result.push_back(dpdy);
    return result;
};

Eigen::VectorXd DifferentialOperators::divergenceOfVectorField(
        const Eigen::VectorXd& u_extended_to_left_right,
        const Eigen::VectorXd& v_extended_to_top_bottom,
        const int    nx,
        const int    ny,
        const double dx,
        const double dy
) {
    const int nx_u{nx + 1};
    const int nx_v{nx};
    Eigen::VectorXd result = Eigen::VectorXd::Zero(nx * ny);

    for (int i = 0; i < (nx * ny); ++i) {
        int ix = i % nx;
        int iy = i / nx;
        // i = ix + nx * iy by definition
        result[i]
        = (
              u_extended_to_left_right[nx_u * iy + ix + 1]
            - u_extended_to_left_right[nx_u * iy + ix]
        ) / dx
        + (
              v_extended_to_top_bottom[nx_v * iy + ix + nx_v]
            - v_extended_to_top_bottom[nx_v * iy + ix]
        ) / dy;
    }

    return result;
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