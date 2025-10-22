#ifndef BOUNDARY_FORCING_BUILDER_STAGGERED_HPP
#define BOUNDARY_FORCING_BUILDER_STAGGERED_HPP

#include <cmath>
#include <Eigen/Dense>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

// x : Node centered, y : Cell centered ; For u
class BoundaryForcingBuilderNCxCCy {
public:
    BoundaryForcingBuilderNCxCCy(
        const int    nx,  const int    ny,
        const double dx,  const double dy,
        const double reynolds
    )
        : nx_(nx), ny_(ny), dx_(dx), dy_(dy), reynolds_(reynolds) {}
    ~BoundaryForcingBuilderNCxCCy() = default;
    Eigen::VectorXd getDirichletBoundaryForcingFrom(
        const Eigen::VectorXd& top,  const Eigen::VectorXd& bottom,
        const Eigen::VectorXd& left, const Eigen::VectorXd& right
    );
    void setDirichletBoundaryForcingFromTopBy(   const Eigen::VectorXd& top,    Eigen::VectorXd& bf); // Each boundary condition for u:
    void setNeumannBoundaryForcingFromTopBy(     const Eigen::VectorXd& top,    Eigen::VectorXd& bf); // Each boundary condition for u:
    void setDirichletBoundaryForcingFromBottomBy(const Eigen::VectorXd& bottom, Eigen::VectorXd& bf); // Each boundary condition for u:
    void setNeumannBoundaryForcingFromBottomBy(  const Eigen::VectorXd& bottom, Eigen::VectorXd& bf); // Each boundary condition for u:
    void setDirichletBoundaryForcingFromLeftBy(  const Eigen::VectorXd& left,   Eigen::VectorXd& bf); // Each boundary condition for u:
    void setDirichletBoundaryForcingFromRightBy( const Eigen::VectorXd& right,  Eigen::VectorXd& bf); // Each boundary condition for u:
    
private:
    const int    nx_;
    const int    ny_;
    const double dx_;
    const double dy_;
    const double reynolds_;
};

// x : Cell centered, y : Node centered ; For v
class BoundaryForcingBuilderCCxNCy {
public:
    BoundaryForcingBuilderCCxNCy(
        const int    nx,  const int    ny,
        const double dx,  const double dy,
        const double reynolds
    )
        : nx_(nx), ny_(ny), dx_(dx), dy_(dy), reynolds_(reynolds) {}
    ~BoundaryForcingBuilderCCxNCy() = default;
    Eigen::VectorXd getDirichletBoundaryForcingFrom(
            const Eigen::VectorXd& top,  const Eigen::VectorXd& bottom,
            const Eigen::VectorXd& left, const Eigen::VectorXd& right
    );
    void setDirichletBoundaryForcingFromTopBy(   const Eigen::VectorXd& top,    Eigen::VectorXd& bf); // Each boundary condition for v:
    void setDirichletBoundaryForcingFromBottomBy(const Eigen::VectorXd& bottom, Eigen::VectorXd& bf); // Each boundary condition for v:
    void setDirichletBoundaryForcingFromLeftBy(  const Eigen::VectorXd& left,   Eigen::VectorXd& bf); // Each boundary condition for v:
    void setNeumannBoundaryForcingFromLeftBy(    const Eigen::VectorXd& left,   Eigen::VectorXd& bf); // Each boundary condition for v:
    void setDirichletBoundaryForcingFromRightBy( const Eigen::VectorXd& right,  Eigen::VectorXd& bf); // Each boundary condition for v:
    void setNeumannBoundaryForcingFromRightBy(   const Eigen::VectorXd& right,  Eigen::VectorXd& bf); // Each boundary condition for v:

private:
    const int    nx_;
    const int    ny_;
    const double dx_;
    const double dy_;
    const double reynolds_;
};

#endif //BOUNDARY_FORCING_BUILDER_STAGGERED_HPP