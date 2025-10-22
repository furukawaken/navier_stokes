#include <cmath>
#include <cstdio>
#include <fstream>
#include <execution>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>
#include <iostream>
#include <memory>
#include <netcdf.h>
#include <stdexcept>
#include <vector>
#include <utility>
#include <omp.h>

// Global variables
#include "../src/Globals.hpp"

// Tests
#include "../test/test_poisson.hpp"
#include "../test/test_heat.hpp"
//#include "../test/test_stokes.hpp"
#include "../test/test_ks.hpp"
#include "../test/test_ns_periodic_smac.hpp"
#include "../test/test_ns_periodic_euler.hpp"
#include "../test/test_ns_periodic_rk4.hpp"
#include "../test/test_ns_periodic_da.hpp"

void callFunctionById(const std::string& id) {
    // Map functions:
    static std::map<std::string, std::function<void()>> functionMap = {
        {"test1", testLinearPoissonDirichlet},
        {"test2", testLinearPoissonNeumann},
        {"test3", testLinearHeatMixedDirichletNeumann},
        {"test4", testKellerSegelNeumann},
        {"test5", testNavierStokesPeriodicSMAC},
        {"test6", testNavierStokesPeriodicEuler},
        {"test7", testNavierStokesPeriodicRK4},
        {"test8", testNavierStokesPeriodicDA},
    };

    auto it = functionMap.find(id);
    if (it != functionMap.end()) {
        it->second(); // Call test
    } else {
        std::cerr << "Function with id " << id << " not found" << std::endl;
    }
}

int main(int argc, char *argv[]) {
    // Machine settings: max is 10.
    omp_set_num_threads(std::min(omp_get_max_threads(), 8)); //Confirm by "sysctl -n hw.logicalcpu_max"
    Eigen::setNbThreads(8);                                  // Eigen OMP setting
    Eigen::initParallel();                                   // Initialize Eigen parallel.

    //callFunctionById("test1");
    //callFunctionById("test2");
    //callFunctionById("test3");
    //callFunctionById("test4");
    //callFunctionById("test5");
    //callFunctionById("test6");
    //callFunctionById("test7");
    callFunctionById("test8");
    return 0;
}
