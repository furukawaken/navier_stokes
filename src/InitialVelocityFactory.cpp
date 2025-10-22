#include "InitialVelocityFactory.hpp"
#include <fstream>
#include <utility>

inline int index(int i, int j, int nx){ return j * nx + i; }

void InitialVelocityAssembler::taylorGreenFlow(
    const int    nx, const int    ny,
    const double dx, const double dy,
    const double reynolds,
    const double xmode, const double ymode,
    Eigen::VectorXd& u,
    Eigen::VectorXd& v,
    Eigen::VectorXd& p,
    const double t
) {
    const double lx = nx * dx;
    const double ly = ny * dy;
    const double kx = 2.0 * M_PI * xmode / lx;
    const double ky = 2.0 * M_PI * ymode / ly;
    const double k2 = kx * kx + ky * ky;
    //const double f  = std::exp(- 2.0 * k2 * t / reynolds);
    const double f  = std::exp(- k2 * t / reynolds);
    const double f2 = f * f; // For p

    for (int j = 0; j < ny; ++j){
        const double y_u = (j + 0.5) * dy;
        const double y_v = j * dy;
        const double y_c = (j + 0.5) * dy;
        for (int i = 0; i < nx; ++i){
            const double x_u = i * dx;
            const double x_v = (i + 0.5) * dx;
            const double x_c = (i + 0.5) * dx;
            const int id = index(i, j, nx);

            u[id] = + std::sin(kx * x_u) * std::cos(ky * y_u) * f;
            v[id] = - std::cos(kx * x_v) * std::sin(ky * y_v) * f;
            p[id] = 0.25 * (
                 std::cos(2.0 * kx * x_c)
               + std::cos(2.0 * ky * y_c)
            ) * f2;
            //(*p)[id] = 0.25*U0*U0 *
            //               ( std::cos(2.0*kx*x_c) + std::cos(2.0*ky*y_c) ) * a_p;
        }
    }
    p.array() -= p.mean();
}