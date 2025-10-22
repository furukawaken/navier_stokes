#include "SteadyForcingFactory.hpp"

/*---SteadyForcingFactory---*/
Eigen::VectorXd constantFunction1D(const int nx, const double forcing_amplitude) {
    Eigen::VectorXd forcing = Eigen::VectorXd::Zero(nx);
    for (int ix = 0; ix < nx; ++ix) {
        forcing[ix] = forcing_amplitude;
    }
    return forcing;
}

Eigen::VectorXd constantFunction2D(const int nx, const int ny, const double forcing_amplitude) {
    Eigen::VectorXd forcing = Eigen::VectorXd::Zero(nx * ny);
    for (int ix = 0; ix < nx; ++ix) {
        for (int jy = 0; jy < ny; ++jy) {
            forcing[jy * nx + ix] = forcing_amplitude;
        }
    }

    return forcing;
}

Eigen::VectorXd sineSine(
    const int    nx, const int    ny,
    const double dx, const double dy
) {
    Eigen::VectorXd result = Eigen::VectorXd::Zero(nx * ny);

    double x_length{nx * dx};
    double y_length{ny * dy};
    double x_center{x_length * 0.5};
    double y_center{y_length * 0.5};
    
    for (int jy=0; jy < ny; jy++) {
        for (int ix=0; ix < nx; ix++) {
            result[ix + nx * jy]
            = (
                std::sin(2.0 * M_PI * (dx * 0.5 + ix * dx - x_center) / x_length)
            )
            * (
                std::sin(2.0 * M_PI * (dy * 0.5 + jy * dy - y_center) / y_length)
            );
        }
    }
    return result;
}
