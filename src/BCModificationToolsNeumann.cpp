#include "BCModificationToolsNeumann.hpp"

void BCModificationToolsNeumann::pinMatrix(Eigen::SparseMatrix<double>& laplacian, const int nx, const int ny) {
    laplacian.coeffRef(0, 0) = 1.0;
    for (int i = 1; i < nx * ny; ++i) {
        laplacian.coeffRef(0, i) = 0.0;
        laplacian.coeffRef(i, 0) = 0.0;
    }
}

Eigen::VectorXd BCModificationToolsNeumann::getPinedVector(const Eigen::VectorXd& u) {
    Eigen::VectorXd u_pinned(u);
    u_pinned[0] = 0.0;
    return u_pinned;
}