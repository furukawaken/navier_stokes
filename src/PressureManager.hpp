#ifndef PRESSURE_MANAGER_HPP
#define PRESSURE_MANAGER_HPP

#include <execution>
#include <Eigen/Dense>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
#include <utility>

#include "BoundaryConditionChecker.hpp"
#include "BoundaryForcingBuilderCellCentered.hpp"
#include "BoundaryOperations.hpp"
#include "BCModificationToolsNeumann.hpp"
#include "TripletListFactory.hpp"

class PressureManagerSMAC {
public:
    using SpMat  = Eigen::SparseMatrix<double>;
    using Solver = Eigen::SimplicialLLT<SpMat>;
    using List   = std::vector<Eigen::Triplet<double>>;
    PressureManagerSMAC(
        const int nx, const int ny,
        const double dx, const double dy, const double dt,
        const List& positive_laplacian_triplet_list
    );
    ~PressureManagerSMAC() = default;
    void setNabla(
        Eigen::VectorXd& dpdx,
        Eigen::VectorXd& dpdy,
        const Eigen::VectorXd& p
    );
    void build(const Eigen::VectorXd& u, const Eigen::VectorXd& v);
    void initialize(const Eigen::VectorXd& u, const Eigen::VectorXd& v);
    void modifyVectorField(Eigen::VectorXd& u, Eigen::VectorXd& v) const;
    const Eigen::VectorXd& getP() const { return *p_; };

private:
    const int nx_;
    const int ny_;
    const double dx_;
    const double dy_;
    const double dt_;

    std::unique_ptr<Eigen::VectorXd> p_;
    std::unique_ptr<Eigen::VectorXd> dpdx_;
    std::unique_ptr<Eigen::VectorXd> dpdy_;
    std::unique_ptr<Eigen::VectorXd> ddpdx_;
    std::unique_ptr<Eigen::VectorXd> ddpdy_;
    std::unique_ptr<Solver> p_solver_;
};

class PressureManagerProjection {
public:
    using SpMat  = Eigen::SparseMatrix<double>;
    using Solver = Eigen::SimplicialLLT<SpMat>;
    using List   = std::vector<Eigen::Triplet<double>>;
    PressureManagerProjection(
        const int nx, const int ny,
        const double dx, const double dy, const double dt,
        const List& positive_laplacian_triplet_list
    );
    ~PressureManagerProjection() = default;
    void setNabla(
        Eigen::VectorXd& dpdx,
        Eigen::VectorXd& dpdy,
        const Eigen::VectorXd& p
    );
    void setDivergence(
        Eigen::VectorXd& div_uv,
        const Eigen::VectorXd& u,
        const Eigen::VectorXd& v
    );
    void build(Eigen::VectorXd& forcing_u, Eigen::VectorXd& forcing_v);
    void initialize(const Eigen::VectorXd& u, const Eigen::VectorXd& v);
    const Eigen::VectorXd& getP() const { return *p_; };
    const Eigen::VectorXd& getDpdx() const { return *dpdx_; };
    const Eigen::VectorXd& getDpdy() const { return *dpdy_; };
    //void modifyVectorField(Eigen::VectorXd& u, Eigen::VectorXd& v) const;

private:
    const int nx_;
    const int ny_;
    const double dx_;
    const double dy_;
    const double dt_;

    std::unique_ptr<Eigen::VectorXd> p_;
    std::unique_ptr<Eigen::VectorXd> dpdx_;
    std::unique_ptr<Eigen::VectorXd> dpdy_;
    //std::unique_ptr<Eigen::VectorXd> ddpdx_;
    //std::unique_ptr<Eigen::VectorXd> ddpdy_;
    std::unique_ptr<Solver> p_solver_;
};

#endif // PRESSURE_MANAGER_HPP