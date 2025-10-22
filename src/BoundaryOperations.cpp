#include "BoundaryOperations.hpp"

Eigen::VectorXd Extensions::getExtendedULeftRight(
        const Eigen::VectorXd& u,
        const Eigen::VectorXd& left,
        const Eigen::VectorXd& right,
        const int    nx,
        const int    ny,
        const double dx
) {
    int i_left{0};
    int i_right{0};
    int i_u{0}; // current position of u, e.g.
                // i = 0 -> iu = 0 (left boundary)
                // i = 1 -> iu = 0, (some process), then i_u++
                // i = 2 -> iu = 1, (some process), then i_u++
                // ....
                // i = nx     -> iu = nx - 1, (some process), no increments!!!
                // i = nx + 1 -> iu = nx - 1, (some process), then i_u++
    Eigen::VectorXd result = Eigen::VectorXd::Zero((nx + 2) * ny);
    for (int i = 0; i< ((nx + 2) * ny); i++) {
        if (i % (nx + 2) == 0) {// left
            result[i] = left[i_left];
            i_left++;
        } else if ((i + 1) % (nx + 2) == 0) {// right
            result[i] = right[i_right];
            i_right++;
            i_u++;
        } else {
            result[i] = u[i_u];
            if ((i + 2) % (nx + 2) == 0) {
                continue;
            } else {
                i_u++;
            }
        }
    }
    return result;
}

Eigen::VectorXd Extensions::getExtendedVTopBottom(
        const Eigen::VectorXd& v,
        const Eigen::VectorXd& top,
        const Eigen::VectorXd& bottom,
        const int    nx,
        const int    ny,
        const double dy
) {
    Eigen::VectorXd result = Eigen::VectorXd::Zero(nx * (ny + 2));
    for (int i = - nx; i< (nx * (ny + 1)); i++) {
        if (i < 0) {
            result[i + nx] =  bottom[i + nx];
        } else if (i >= nx * ny) {
            result[i + nx] =  top[i - nx * ny];
        } else {
            result[i + nx] = v[i];
        }
    }
    return result;
}

void BoundaryValueHandler::setNoSlip(
    std::vector<Eigen::VectorXd>& uv,
    const int nx_u, const int ny_u,
    const int nx_v, const int ny_v
) {
    // For u:
    for (int ix =0; ix < nx_u; ix++) {
        uv[0][ix]                     = 0.0;
        uv[0][ix + nx_u * (ny_u - 1)] = 0.0;
    }
    // For v:
    for (int jy = 0; jy < ny_v; jy++) {
        uv[1][ jy      * nx_v    ] = 0.0;
        uv[1][(jy + 1) * nx_v - 1] = 0.0;
    }
}