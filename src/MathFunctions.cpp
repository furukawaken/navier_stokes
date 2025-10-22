#include "MathFunctions.hpp"


std::vector<Eigen::VectorXd> MathFunctions::oseenVortex(
    const int nx, const int ny,
    const double dx, const double dy,
    const double scale, double circulation
) {
    Eigen::VectorXd u{Eigen::VectorXd::Zero((nx - 1) *  ny     )}; 
    Eigen::VectorXd v{Eigen::VectorXd::Zero(nx       * (ny - 1))}; 
    std::vector<Eigen::VectorXd> result{};

    for (int ix=0; ix < nx-1; ix++){
        for (int jy; jy < ny; jy++){
            if ((ix==0) && (jy==0)) {
                u[ix + jy * nx] = 0.0;
            } else{
                double xu{dx * ix};
                double yu{(dy / 2) + dy * jy};
                double squired_norm = xu * xu + yu * yu;
                double amplitude = circulation * (
                    1
                    - std::exp(
                        - squired_norm / (4 * scale)
                    )) / (2 * M_PI * squired_norm);
                u[ix + jy * nx] = - yu  * amplitude; 
            }
        }
    }

    for (int ix=0; ix < nx; ix++){
        for (int jy; jy < ny-1; jy++){
            if ((ix==0) && (jy==0)) {
                v[ix + jy * nx] = 0.0;
            } else{
                double xv{dx/2 + dx * ix};
                double yv{dy * jy};
                double squired_norm = xv * xv + yv * yv;
                double amplitude = circulation * (
                    1
                    - std::exp(
                        - squired_norm / (4 * scale)
                    )) / (2 * M_PI * squired_norm);
                v[ix + jy * nx] = xv * amplitude; 
            }
        }
    }

    result.push_back(u);
    result.push_back(v);
    return result;
}

std::vector<Eigen::VectorXd> MathFunctions::oseenVortexCentered(
    const int nx, const int ny,
    const double dx, const double dy,
    const double scale, double circulation
) {
    Eigen::VectorXd u{Eigen::VectorXd::Zero((nx - 1) *  ny     )}; 
    Eigen::VectorXd v{Eigen::VectorXd::Zero(nx       * (ny - 1))};
    double center_x{nx * dx * 0.5};
    double center_y{ny * dy * 0.5};
    std::vector<Eigen::VectorXd> result{};

    for (int ix=0; ix < nx-1; ix++){
        for (int jy; jy < ny; jy++){
            if ((ix==0) && (jy==0)) {
                u[ix + jy * nx] = 0.0;
            } else{
                double xu{           dx * ix - center_x};
                double yu{(dy / 2) + dy * jy - center_y};
                double squired_norm = xu * xu + yu * yu;
                double amplitude = circulation * (
                    1
                    - std::exp(
                        - squired_norm / (4 * scale)
                    )) / (2 * M_PI * squired_norm);
                u[ix + jy * nx] = - yu  * amplitude; 
            }
        }
    }

    for (int ix=0; ix < nx; ix++){
        for (int jy; jy < ny-1; jy++){
            if ((ix==0) && (jy==0)) {
                v[ix + jy * nx] = 0.0;
            } else{
                double xv{dx/2 + dx * ix - center_x};
                double yv{       dy * jy - center_y};
                double squired_norm = xv * xv + yv * yv;
                double amplitude = circulation * (
                    1
                    - std::exp(
                        - squired_norm / (4 * scale)
                    )) / (2 * M_PI * squired_norm);
                v[ix + jy * nx] = xv * amplitude; 
            }
        }
    }

    result.push_back(u);
    result.push_back(v);
    return result;
};