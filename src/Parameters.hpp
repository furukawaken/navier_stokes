#ifndef PARAMETERS_HPP
#define PARAMETERS_HPP

#include <iostream>
#include <iomanip>
#include <cstdlib>

class Parameters {
public:

    Parameters() = default;
    Parameters(const Parameters& other) = default;

    //Physical:
    // (CFL = nu * dt / dx^2 + dy^2 <= 0.25)
    int nx = 16 + 0;
    int ny = 16 + 1;
    int nt = 1000 + 1;
    double dx = 0.0625;
    double dy = 0.0625;
    double dt = 0.01;
    double diffusivity   = 0.1;
    //double diffusivity_x = 0.1;
    //double diffusivity_y = 0.1;
    double reynolds = 10.0;
    double forcing_amplitude = 1.0;

    // Numerical settings
    int saving_interval = 10;
    double sor_omega = 1.5;

    void show() {
        std::cout << std::setw(5)         << "nx                : " << nx                << std::endl;
        std::cout << std::setw(5)         << "ny                : " << ny                << std::endl;
        std::cout << std::setw(5)         << "nt                : " << nt                << std::endl;
        std::cout << std::setprecision(7) << "dx                : " << dx                << std::endl;
        std::cout << std::setprecision(7) << "dy                : " << dy                << std::endl;
        std::cout << std::setprecision(7) << "dt                : " << dt                << std::endl;
        std::cout << std::setprecision(7) << "diffusivity       : " << diffusivity       << std::endl;
        //std::cout << std::setprecision(7) << "diffusivity_x     : " << diffusivity_x     << std::endl;
        //std::cout << std::setprecision(7) << "diffusivity_y     : " << diffusivity_y     << std::endl;
        std::cout << std::setprecision(7) << "reynolds          : " << reynolds          << std::endl;
        std::cout << std::setprecision(7) << "forcing_amplitude : " << forcing_amplitude << std::endl;
        std::cout << std::setprecision(7) << "saving_interval   : " << saving_interval   << std::endl;
        std::cout << std::setprecision(7) << "sor_omega         : " << sor_omega         << std::endl;
        
        // CFL Value
        std::cout << std::setprecision(7)
                 << "cfl               : "
                 << (diffusivity * dt / ( dx * dx + dy * dy))
                 << std::endl;
    }

    void checkStokesCFL() {
        double cfl = dt / (reynolds * ( dx * dx + dy * dy));
        if (cfl < 0.25) {
            std::cout << "CFL = " << cfl << ". (OK)" << std::endl;
            if (cfl > 0.1) {
                std::cout << "However, CFL should be less than 0.1 for stability." << std::endl;
            }
        } else {
            std::cout << "--------------------------------------------------"
                      << std::endl;
            std::cout << "CFL = " << cfl << ". ";
            std::cout << "Then, the computations are STOPPED!!!" << std::endl;
            std::cout << "--------------------------------------------------"
                      << std::endl;
            std::exit(0);
        }
    }
};

#endif //PARAMETERS_HPP