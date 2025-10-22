#ifndef BOUNDARY_FORCING_BUILDER_CELL_CENTERED_HPP
#define BOUNDARY_FORCING_BUILDER_CELL_CENTERED_HPP

#include <cmath>
#include <Eigen/Dense>
#include <utility>
#include <vector>

#include "BoundaryConditionChecker.hpp"

class BoundaryForcingBuilderCellCentered {
public:
    BoundaryForcingBuilderCellCentered(
        const int    nx,  const int    ny,
        const double dx,  const double dy,
        const double nu,
        const BoundaryConditionChecker& bc_checker
    );
    ~BoundaryForcingBuilderCellCentered() = default;
    Eigen::VectorXd getBoundaryForcingFrom(
            const Eigen::VectorXd& top,
            const Eigen::VectorXd& bottom,
            const Eigen::VectorXd& left,
            const Eigen::VectorXd& right
    );

private:
    void setBoundaryForcingFromTopBy(   const Eigen::VectorXd& top,    Eigen::VectorXd& bf);
    void setBoundaryForcingFromBottomBy(const Eigen::VectorXd& bottom, Eigen::VectorXd& bf);
    void setBoundaryForcingFromLeftBy(  const Eigen::VectorXd& left,   Eigen::VectorXd& bf);
    void setBoundaryForcingFromRightBy( const Eigen::VectorXd& right,  Eigen::VectorXd& bf);

    const int    nx_;
    const int    ny_;
    const double dx_;
    const double dy_;
    const double nu_;
    std::unique_ptr<BoundaryConditionChecker> bc_checker_;
};

#endif //BOUNDARY_FORCING_BUILDER_CELL_CENTERED_HPP