#include "InitialFunctionFactory.hpp"

Eigen::VectorXd InitialFunctionFactory::cosHill(
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
                0.50 *
                (
                      std::cos(2.0 * M_PI * (  dx * 0.5 + ix        * dx - x_center) / x_length)
                    + std::cos(2.0 * M_PI * (- dx * 0.5 + (nx - ix) * dx - x_center) / x_length) // For symmetric
                )
                + 1.0
            )
            * (
                0.50 *
                (
                      std::cos(2.0 * M_PI * (  dy * 0.5 + jy        * dy - y_center) / y_length)
                    + std::cos(2.0 * M_PI * (- dy * 0.5 + (ny - jy) * dy - y_center) / y_length) // For symmetric
                )
                + 1.0
            );
        }
    }
    return result;
}

Eigen::VectorXd InitialFunctionFactory::twinCosHill(
    const int    nx, const int    ny,
    const double dx, const double dy
) {
    Eigen::VectorXd result = Eigen::VectorXd::Zero(nx * ny);

    double x_length{nx * dx};
    double y_length{ny * dy};
    double x_diam{x_length * 0.5};
    double y_diam{y_length * 0.5};
    double x_rad{x_length * 0.25};
    double y_rad{y_length * 0.25};
    //double x_center1{x_length * 0.25};
    //double y_center1{y_length * 0.25};
    //double x_center2{x_length * 0.75};
    //double y_center2{y_length * 0.75};
    double x_center1{x_length * 0.4};
    double y_center1{y_length * 0.4};
    double x_center2{x_length * 0.6};
    double y_center2{y_length * 0.6};

    auto cosBump = [](double x, double y, double xc, double yc, double lx, double ly) -> double {
        double val_x = std::cos(2.0 * M_PI * (x - xc) / lx);
        double val_y = std::cos(2.0 * M_PI * (y - yc) / ly);
        return (val_x + 1.0) * (val_y + 1.0);  // always ≥ 0
    };
    for (int jy = 0; jy < ny; ++jy) {
        for (int ix = 0; ix < nx; ++ix) {
            double x = dx * (ix + 0.5);
            double y = dy * (jy + 0.5);
            double h1 = (
                   (abs(x - x_center1) < x_rad)
                && (abs(y - y_center1) < y_rad)
            )?
                cosBump(x, y, x_center1, y_center1, x_diam, y_diam)
                : 0.0;
            double h2 = (
                   (abs(x - x_center2) < x_rad)
                && (abs(y - y_center2) < y_rad)
            )?
                cosBump(x, y, x_center2, y_center2, x_diam, y_diam)
                : 0.0;
            result[ix + nx * jy] = h1 + h2;
        }
    }
    return result;
}

Eigen::VectorXd InitialFunctionFactory::debugData(
    const int    nx, const int    ny,
    const double dx, const double dy
) {
    double x_length{nx * dx};
    double y_length{ny * dy};
    double x_center{x_length * 0.5};
    double y_center{y_length * 0.5};
    Eigen::VectorXd result = Eigen::VectorXd::Zero(nx * ny);
    for (int jy=0; jy < ny; jy++) {
        for (int ix=0; ix < nx; ix++) {
            if (jy < ny * 0.5) {
                result[jy * nx + ix] = 1.0;
            } else {
                result[jy * nx + ix] = 0.0;
            }
        }
    }
    return result;
}