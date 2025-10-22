#include "BoundaryForcingBuilderCellCentered.hpp"

/*---BoundaryForcingBuilderCellCentered---*/
BoundaryForcingBuilderCellCentered::BoundaryForcingBuilderCellCentered(
        const int    nx,  const int    ny,
        const double dx,  const double dy,
        const double nu,
        const BoundaryConditionChecker& bc_checker
    )
        : nx_(nx), ny_(ny), dx_(dx), dy_(dy), nu_(nu), bc_checker_(std::make_unique<BoundaryConditionChecker>(bc_checker)) {}

Eigen::VectorXd BoundaryForcingBuilderCellCentered::getBoundaryForcingFrom(
        const Eigen::VectorXd& top,
        const Eigen::VectorXd& bottom,
        const Eigen::VectorXd& left,
        const Eigen::VectorXd& right
) {
    Eigen::VectorXd bf = Eigen::VectorXd::Zero(nx_ * ny_);
    setBoundaryForcingFromTopBy(   top,    bf);
    setBoundaryForcingFromBottomBy(bottom, bf);
    setBoundaryForcingFromLeftBy(  left,   bf);
    setBoundaryForcingFromRightBy( right,  bf);
    return bf;
}

// Each boundary condition:
void BoundaryForcingBuilderCellCentered::setBoundaryForcingFromTopBy(const Eigen::VectorXd& top, Eigen::VectorXd& bf) {
    assert(top.size() == nx_);

    const double dx2{dx_ * dx_};
    const double dy2{dy_ * dy_};
    for (int ix = 0; ix < nx_; ++ix) {
        if (bc_checker_->isDirichlet("t")) {
            bf[nx_ * (ny_ - 1) + ix] += 2.0 * nu_ * top[ix] / dy2;
        } else if (bc_checker_->isNeumann("t")) {
            bf[nx_ * (ny_ - 1) + ix] += nu_ * top[ix] / dy_;
        } else {
            std::cerr << "Error: Invalid boundary condition for 't' at index "
                      << "ix= " << ix << std::endl;
            throw std::invalid_argument("Invalid boundary condition");
        }
    }
}

void BoundaryForcingBuilderCellCentered::setBoundaryForcingFromBottomBy(const Eigen::VectorXd& bottom, Eigen::VectorXd& bf) {
    assert(bottom.size() == nx_);

    const double dx2{dx_ * dx_};
    const double dy2{dy_ * dy_};
    for (int ix = 0; ix < nx_; ++ix) {
        if (bc_checker_->isDirichlet("b")) {
            bf[ix] += 2.0 * nu_ * bottom[ix] / dy2;
        } else if (bc_checker_->isNeumann("b")) {
            bf[ix] += nu_ * bottom[ix] / dy_;
        } else {
            std::cerr << "Error: Invalid boundary condition for 'b' at index "
                      << "ix= " << ix << std::endl;
            throw std::invalid_argument("Invalid boundary condition");
        }
    }
}

void BoundaryForcingBuilderCellCentered::setBoundaryForcingFromLeftBy(const Eigen::VectorXd& left, Eigen::VectorXd& bf) {
    assert(left.size() == ny_);
    
    const double dx2{dx_ * dx_};
    const double dy2{dy_ * dy_};
    for (int jy = 0; jy < ny_; ++jy) {
        if (bc_checker_->isDirichlet("l")) {
            bf[jy * nx_] += 2.0 * nu_ * left[jy] / dx2;
        } else if (bc_checker_->isNeumann("l")) {
            bf[jy * nx_] += nu_ * left[jy] / dx_;
        } else {
            std::cerr << "Error: Invalid boundary condition for 'l' at index "
                      << "jy= " << jy << std::endl;
            throw std::invalid_argument("Invalid boundary condition");
        }
    } 
}

void BoundaryForcingBuilderCellCentered::setBoundaryForcingFromRightBy(const Eigen::VectorXd& right, Eigen::VectorXd& bf) {
    assert(right.size() == ny_);

    const double dx2{dx_ * dx_};
    const double dy2{dy_ * dy_};
    for (int jy = 0; jy < ny_; ++jy) {
        if (bc_checker_->isDirichlet("r")) {
            bf[(jy + 1) * nx_ - 1] += 2.0 * nu_ * right[jy] / dx2;
        } else if (bc_checker_->isNeumann("r")) {
            bf[(jy + 1) * nx_ - 1] += nu_ * right[jy] / dx_;
        } else {
            std::cerr << "Error: Invalid boundary condition for 'r' at index "
                      << "jy= " << jy << std::endl;
            throw std::invalid_argument("Invalid boundary condition");
        }
    }
}