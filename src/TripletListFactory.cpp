#include "TripletListFactory.hpp"


/* LaplacianTripletFactory :
    - \Delta u = f
    i.e. Laplacian (- \Delta) is positive.
    
    Note : FOR CELL-CENTERED FIELD
*/
//std::vector<Eigen::Triplet<double>> LaplacianTripletFactory::getList(
std::vector<Eigen::Triplet<double>> getMixedDirichletNeumannList(
        const int    nx,  const int    ny,
        const double dx,  const double dy,
        const double nu,
        const BoundaryConditionChecker& bc_checker
) {
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(5 * nx * ny);

    const double dx2 = dx * dx;
    const double dy2 = dy * dy;
    const double cx  = nu / dx2;
    const double cy  = nu / dy2;

    auto index = [nx](int ix, int jy) { return jy * nx + ix; };

    for (int jy = 0; jy < ny; ++jy) {
        for (int ix = 0; ix < nx; ++ix) {
            int i = index(ix, jy);
            double center = 2.0 * (cx + cy);

            // Left
            if (ix > 0) {
                triplets.emplace_back(i, index(ix - 1, jy), -cx);
            } else {
                if (bc_checker.isDirichlet("l")) {
                    center += cx;
                } else if (bc_checker.isNeumann("l")) {
                    center -= cx;;
                } else {
                    throw std::invalid_argument("Invalid boundary condition: 'l'");
                }
            }

            // Right
            if (ix < nx - 1) {
                triplets.emplace_back(i, index(ix + 1, jy), -cx);
            } else {
                if (bc_checker.isDirichlet("r")) {
                    center += cx;
                } else if (bc_checker.isNeumann("r")) {
                    center -= cx;;
                } else {
                    throw std::invalid_argument("Invalid boundary condition: 'r'");
                }
            }

            // Bottom
            if (jy > 0) {
                triplets.emplace_back(i, index(ix, jy - 1), -cy);
            } else {
                if (bc_checker.isDirichlet("b")) {
                    center += cy;
                } else if (bc_checker.isNeumann("b")) {
                    center -= cy;
                } else {
                    throw std::invalid_argument("Invalid boundary condition: 'b'");
                }
            }

            // Top
            if (jy < ny - 1) {
                triplets.emplace_back(i, index(ix, jy + 1), -cy);
            } else {
                if (bc_checker.isDirichlet("t")) {
                    center += cy;
                } else if (bc_checker.isNeumann("t")) {
                    center -= cy;
                } else {
                    throw std::invalid_argument("Invalid boundary condition: 't'");
                }
            }
            // Center (i, i)
            triplets.emplace_back(i, i, center);
        }
    }

    return triplets;
}

std::vector<Eigen::Triplet<double>> getPurePeriodicList(
        const int    nx,  const int    ny,
        const double dx,  const double dy,
        const double nu
) {
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(5 * nx * ny);

    const double dx2 = dx * dx;
    const double dy2 = dy * dy;
    const double cx  = nu / dx2;
    const double cy  = nu / dy2;

    auto index = [nx, ny](int ix, int jy) {
        int x = (nx + ix) % nx;
        int y = (ny + jy) % ny;
        return y * nx + x;
    };

    for (int jy = 0; jy < ny; ++jy) {
        for (int ix = 0; ix < nx; ++ix) {
            int i = index(ix, jy);
            double center_val = 2.0 * (cx + cy);
            triplets.emplace_back(i, i,                     center_val);
            triplets.emplace_back(i, index(ix - 1, jy),     - cx);
            triplets.emplace_back(i, index(ix + 1, jy),     - cx);
            triplets.emplace_back(i, index(ix,     jy - 1), - cy);
            triplets.emplace_back(i, index(ix,     jy + 1), - cy);
        }
    }

    return triplets;
}

std::vector<Eigen::Triplet<double>> LaplacianTripletFactory::getList(
        const int    nx,  const int    ny,
        const double dx,  const double dy,
        const double nu,
        const BoundaryConditionChecker& bc_checker
) {
    if (bc_checker.isMixedDirichletNeumann()) {
        return getMixedDirichletNeumannList(
            nx,  ny,
            dx,  dy,
            nu,
            bc_checker
        );
    }

    if(bc_checker.isPurePeriodic()) {
        return getPurePeriodicList(
            nx,  ny,
            dx,  dy,
            nu
        );
    }
    
    throw std::invalid_argument(
        std::string("Unsupported boundary combination: ") + bc_checker.describe() +
        " (supported: all-Dirichlet/Neumann-only or pure-periodic)"
    );
}

/*---InhomogeneousHeatTripletFactory---*/
std::vector<Eigen::Triplet<double>> InhomogeneousHeatTripletFactory::getEvolutionMatrixList(
        const int    nx,  const int    ny,
        const double dx,  const double dy,
        const double nu,
        const BoundaryConditionChecker& bc_checker
) {
    std::vector<Eigen::Triplet<double>> triplet_list(5 * nx * ny);
    const double dx2{dx * dx};                  // dx^2 
    const double dy2{dy * dy};                  // dy^2 
    double it{0.0};
    double ib{0.0};
    double il{0.0};
    double ir{0.0};
    for (int i = 0; i < (nx * ny); ++i) {
        it = 0.0;
        ib = 0.0;
        il = 0.0;
        ir = 0.0;
        if (i < (nx * (ny - 1))) {
            // No top point
            triplet_list.emplace_back(
                i,
                i + nx,
                nu / dy2
            );
        } else {
            // A top point
            if (bc_checker.isDirichlet("t")) {
                it = - nu / dy2;
            } else if (bc_checker.isNeumann("t")) {
                it =   nu / dy2;
            } else {
                std::cerr << "Error: Invalid boundary condition for 't' at index "
                          << i << std::endl;
                throw std::invalid_argument("Invalid boundary condition");
            }
        }
        if (i >= nx) {
            // No bottom point
            triplet_list.emplace_back(
                i,
                i - nx,
                nu / dy2
            );
        } else {
            // A bottom point
            if (bc_checker.isDirichlet("b")) {
                ib = - nu / dy2;
            } else if (bc_checker.isNeumann("b")) {
                ib =   nu / dy2;
            } else {
                std::cerr << "Error: Invalid boundary condition for 'b' at index "
                          << i << std::endl;
                throw std::invalid_argument("Invalid boundary condition");
            }
        }
        if ((i % nx) != 0) {
            // No left point
            triplet_list.emplace_back(
                i,
                i - 1,
                nu / dx2
            );
        } else {
            // A left point
            if (bc_checker.isDirichlet("l")) {
                il = - nu / dx2;
            } else if (bc_checker.isNeumann("l")) {
                il =   nu / dx2;
            } else {
                std::cerr << "Error: Invalid boundary condition for 'l' at index "
                          << i << std::endl;
                throw std::invalid_argument("Invalid boundary condition");
            }
        }
        if (((i + 1) % nx) != 0) {
            // No right point
            triplet_list.emplace_back(
                i,
                i + 1,
                nu / dx2
            );
        } else {
            // A right point
            if (bc_checker.isDirichlet("r")) {
                ir = - nu / dx2;
            } else if (bc_checker.isNeumann("r")) {
                ir =   nu / dx2;
            } else {
                std::cerr << "Error: Invalid boundary condition for 'r' at index "
                          << i << std::endl;
                throw std::invalid_argument("Invalid boundary condition");
            }
        }
        triplet_list.emplace_back(
            i,
            i,
              il + ir - 2 * nu / dx2
            + ib + it - 2 * nu / dy2
        );//Center point
    }
    return triplet_list;
}

///* The positions of boundary of u (x: Node Centered, y: Cell Centered):
//    Top    : on an artificial point, i.e. (( , ny-1) + (, ny)) / 2
//    Bottom : on an artificial point, i.e. (( , -1) + (, 0)) / 2
//    Left   : on a boundary
//    Right  : on a boundary
//*/
std::vector<Eigen::Triplet<double>> HorizontalVelocityTripletFactory::getDirichletList(
        const int    nx,
        const int    ny,
        const double dx,
        const double dy,
        const double reynolds
) {
    std::vector<Eigen::Triplet<double>> triplet_list;
    const double dx2{dx * dx};           // dx^2 
    const double dy2{dy * dy};           // dy^2
    const double re_dx2{reynolds * dx2};
    const double re_dy2{reynolds * dy2}; 
    double it{0.0};
    double ib{0.0};
    double il{0.0};
    double ir{0.0};
    for (int i = 0; i < (nx * ny); ++i) {
        it = 0.0;
        ib = 0.0;
        il = 0.0;
        ir = 0.0;
        if (i >= nx) {
            triplet_list.emplace_back(
                i,
                i - nx,
                1.0 / re_dy2
            );//No lower(bottom) point
        } else {
            ib = - 1.0 / re_dy2;
        }
        if (i < (nx * (ny - 1))) {
            triplet_list.emplace_back(
                i,
                i + nx,
                1.0 / re_dy2
            );//No top point
        } else {
            it = - 1.0 / re_dy2;
        }
        if ((i % nx) != 0) {
            triplet_list.emplace_back(
                i,
                i - 1,
                1.0 / re_dx2
            );//No left point
        } else {
            il = 0.0;
        }
        if (((i + 1) % nx) != 0) {
            triplet_list.emplace_back(
                i,
                i + 1,
                1.0 / re_dx2
            );//No right point
        } else {
            ir = 0.0;
        }
        // Diagonal parts
        triplet_list.emplace_back(
            i,
            i,
              il + ir
            + ib + it
            - 2.0 / re_dx2
            - 2.0 / re_dy2
        );
    }
    return triplet_list;
}

///* The positions of boundary of v (x: Cell Centered, y: Node Centered)::
//    Top    : on a boundary
//    Bottom : on a boundary
//    Left   : on an artificial point, i.e. ((-1, ) + (0,)) / 2
//    Right  : on an artificial point, i.e. ((nx-1, ) + (ny, )) / 2
//*/
std::vector<Eigen::Triplet<double>> VerticalVelocityTripletFactory::getDirichletList(
        const int    nx,
        const int    ny,
        const double dx,
        const double dy,
        const double reynolds
) {
    std::vector<Eigen::Triplet<double>> triplet_list;
    const double dx2{dx * dx};           // dx^2 
    const double dy2{dy * dy};           // dy^2
    const double re_dx2{reynolds * dx2};
    const double re_dy2{reynolds * dy2};
    double it{0.0};
    double ib{0.0};
    double il{0.0};
    double ir{0.0};
    for (int i = 0; i < (nx * ny); ++i) {
        it = 0.0;
        ib = 0.0;
        il = 0.0;
        ir = 0.0;
        if (i >= nx) {
            triplet_list.emplace_back(
                i,
                i - nx,
                1.0 / re_dy2
            );//No bottom point
        } else {
            ib = 0.0;
        }
        if (i < (nx * (ny - 1))) {
            triplet_list.emplace_back(
                i,
                i + nx,
                1.0 / re_dy2
            );//No top point
        } else {
            it = 0.0;
        }
        if ((i % nx) != 0) {
            triplet_list.emplace_back(
                i,
                i - 1,
                1.0 / re_dx2
            );//No left point
        } else {
            il = - 1.0 / re_dx2;
        }
        if (((i + 1) % nx) != 0) {
            triplet_list.emplace_back(
                i,
                i + 1,
                1.0 / re_dx2
            );//No right point
        } else {
            ir = - 1.0 / re_dx2;
        }
        // Diagonals
        triplet_list.emplace_back(
            i,
            i,
              il + ir
            + ib + it
            - 2.0 / re_dx2
            - 2.0 / re_dy2
        );
    }
    return triplet_list;
}