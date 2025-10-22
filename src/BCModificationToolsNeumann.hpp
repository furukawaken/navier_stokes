#ifndef B_C_MODIFICATION_TOOLS_NEUMANN_HPP
#define B_C_MODIFICATION_TOOLS_NEUMANN_HPP

#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace BCModificationToolsNeumann {
    void pinMatrix(Eigen::SparseMatrix<double>& laplacian, const int nx, const int ny);
    Eigen::VectorXd getPinedVector(const Eigen::VectorXd& u);
}
#endif //B_C_MODIFICATION_TOOLS_NEUMANN_HPP