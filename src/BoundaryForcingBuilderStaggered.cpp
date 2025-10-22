#include "BoundaryForcingBuilderStaggered.hpp"

/*---BoundaryForcingBuilderNCxCCy---*/
Eigen::VectorXd BoundaryForcingBuilderNCxCCy::getDirichletBoundaryForcingFrom(
        const Eigen::VectorXd& top,
        const Eigen::VectorXd& bottom,
        const Eigen::VectorXd& left,
        const Eigen::VectorXd& right
) {
    Eigen::VectorXd boundary_forcing = Eigen::VectorXd::Zero(nx_ * ny_);

    setDirichletBoundaryForcingFromTopBy(   top,    boundary_forcing);
    setDirichletBoundaryForcingFromBottomBy(bottom, boundary_forcing);
    setDirichletBoundaryForcingFromLeftBy(  left,   boundary_forcing);
    setDirichletBoundaryForcingFromRightBy( right,  boundary_forcing);

    return boundary_forcing;
}

// Each boundary condition for u:
void BoundaryForcingBuilderNCxCCy::setDirichletBoundaryForcingFromTopBy(
        const Eigen::VectorXd& top,
              Eigen::VectorXd& boundary_forcing
) {
    assert(top.size() == nx_);

    const double dy2{dy_ * dy_};
    
    for (int ix = 0; ix < nx_; ++ix) {
        boundary_forcing[nx_ * (ny_ - 1) + ix] += 2.0 * reynolds_ * top[ix] / dy2;
    }
}
    
void BoundaryForcingBuilderNCxCCy::setNeumannBoundaryForcingFromTopBy(
        const Eigen::VectorXd& top,
              Eigen::VectorXd& boundary_forcing
) {
    assert(top.size() == nx_);

    const double dy2{dy_ * dy_};

    for (int ix = 0; ix < nx_; ++ix) {
        boundary_forcing[nx_ * (ny_ - 1) + ix] += reynolds_ * top[ix] / dy_;
    }
}

void BoundaryForcingBuilderNCxCCy::setDirichletBoundaryForcingFromBottomBy(
        const Eigen::VectorXd& bottom,
              Eigen::VectorXd& boundary_forcing
) {
    assert(bottom.size() == nx_);

    double dy2{dy_ * dy_};

    for (int ix = 0; ix < nx_; ++ix) {
        boundary_forcing[ix] += 2.0 * reynolds_ * bottom[ix] / dy2;
    }
}

void BoundaryForcingBuilderNCxCCy::setNeumannBoundaryForcingFromBottomBy(
        const Eigen::VectorXd& bottom,
              Eigen::VectorXd& boundary_forcing
) {
    assert(bottom.size() == nx_);
    
    double dy2{dy_ * dy_};

    for (int ix = 0; ix < nx_; ++ix) {
        boundary_forcing[ix] += reynolds_ * bottom[ix] / dy_;
    }
}

void BoundaryForcingBuilderNCxCCy::setDirichletBoundaryForcingFromLeftBy(
        const Eigen::VectorXd& left,
              Eigen::VectorXd& boundary_forcing
) {
    assert(left.size() == ny_);

    double dx2{dx_ * dx_};

    for (int jy = 0; jy < ny_; ++jy) {
        boundary_forcing[jy * nx_] += left[jy] / (reynolds_ * dx2);
    } 
}

void BoundaryForcingBuilderNCxCCy::setDirichletBoundaryForcingFromRightBy(
        const Eigen::VectorXd& right,
              Eigen::VectorXd& boundary_forcing
) {
    assert(right.size() == ny_);

    const double dx2{dx_ * dx_};

    for (int jy = 0; jy < ny_; ++jy) {
        boundary_forcing[(jy + 1) * nx_ - 1] += right[jy] / (reynolds_ * dx2);
    }
}

/*---BoundaryForcingBuilderCCxNCy---*/
Eigen::VectorXd BoundaryForcingBuilderCCxNCy::getDirichletBoundaryForcingFrom(
        const Eigen::VectorXd& top,  const Eigen::VectorXd& bottom,
        const Eigen::VectorXd& left, const Eigen::VectorXd& right
) {
    Eigen::VectorXd boundary_forcing = Eigen::VectorXd::Zero(nx_ * ny_);
    
    setDirichletBoundaryForcingFromTopBy(   top,    boundary_forcing);
    setDirichletBoundaryForcingFromBottomBy(bottom, boundary_forcing);
    setDirichletBoundaryForcingFromLeftBy(  left,   boundary_forcing);
    setDirichletBoundaryForcingFromRightBy( right,  boundary_forcing);

    return boundary_forcing;
}

// Each boundary condition for v:
void BoundaryForcingBuilderCCxNCy::setDirichletBoundaryForcingFromTopBy(
        const Eigen::VectorXd& top,
              Eigen::VectorXd& boundary_forcing
) {
    assert(top.size() == nx_);
    
    const double dy2{dy_ * dy_};

    for (int ix = 0; ix < nx_; ++ix) {
        boundary_forcing[nx_ * (ny_ - 1) + ix] += top[ix] / (reynolds_ * dy2);
    }
}

void BoundaryForcingBuilderCCxNCy::setDirichletBoundaryForcingFromBottomBy(
        const Eigen::VectorXd& bottom,
              Eigen::VectorXd& boundary_forcing
) {
    assert(bottom.size() == nx_);
    
    const double dy2{dy_ * dy_};

    for (int ix = 0; ix < nx_; ++ix) {
        boundary_forcing[ix] += bottom[ix] / (reynolds_ * dy2);
    }
}

void BoundaryForcingBuilderCCxNCy::setDirichletBoundaryForcingFromLeftBy(
        const Eigen::VectorXd& left,
              Eigen::VectorXd& boundary_forcing
) {
    assert(left.size() == ny_);

    const double dx2{dx_ * dx_};

    for (int jy = 0; jy < ny_; ++jy) {
        boundary_forcing[jy * nx_] += 2.0 * reynolds_ * left[jy] / dx2;
    } 
}
    
void BoundaryForcingBuilderCCxNCy::setNeumannBoundaryForcingFromLeftBy(
        const Eigen::VectorXd& left,
              Eigen::VectorXd& boundary_forcing
) {
    assert(left.size() == ny_);

    const double dx2{dx_ * dx_};
    
    for (int jy = 0; jy < ny_; ++jy) {
        boundary_forcing[jy * nx_] += reynolds_ * left[jy] / dx_;
    } 
}

void BoundaryForcingBuilderCCxNCy::setDirichletBoundaryForcingFromRightBy(
        const Eigen::VectorXd& right,
              Eigen::VectorXd& boundary_forcing
) {
    assert(right.size() == ny_);

    const double dx2{dx_ * dx_};

    for (int jy = 0; jy < ny_; ++jy) {
        boundary_forcing[(jy + 1) * nx_ - 1] += 2.0 * reynolds_ * right[jy] / dx2;
    }
}

void BoundaryForcingBuilderCCxNCy::setNeumannBoundaryForcingFromRightBy(
        const Eigen::VectorXd& right,
              Eigen::VectorXd& boundary_forcing
) {
    assert(right.size() == ny_);

    const double dx2{dx_ * dx_};

    for (int jy = 0; jy < ny_; ++jy) {
        boundary_forcing[(jy + 1) * nx_ - 1] += reynolds_ * right[jy] / dx_;
    }
}