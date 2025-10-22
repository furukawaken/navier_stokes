#ifndef BOUNDARY_CONDITION_CHECKER_HPP
#define BOUNDARY_CONDITION_CHECKER_HPP

#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

class BoundaryConditionChecker {
public:
    /*
    BoundaryConditionChecker(
        const std::vector<std::map<std::string,
        std::string>>& bc_list
    )
        : bc_list_(bc_list) {}
    */
    BoundaryConditionChecker(const std::unordered_map<std::string, std::string>& bc_list) : bc_list_(bc_list) {}
    BoundaryConditionChecker(const std::string& d_or_n_or_t);
    BoundaryConditionChecker(const BoundaryConditionChecker& checker) = default;
    ~BoundaryConditionChecker() = default;
    bool isDirichlet(const std::string& side) const { return checkCondition(side, "D"); }
    bool isNeumann(  const std::string& side) const { return checkCondition(side, "N"); }
    bool isPeriodic( const std::string& side) const { return checkCondition(side, "T"); }
    bool isPureDirichlet() const;
    bool isPureNeumann()   const;
    bool isPurePeriodic()  const;
    bool isMixedDirichletNeumann() const;
    std::string describe() const;

private:
    bool checkCondition(const std::string& side, const std::string& d_or_n_or_t) const;

    // Members
    //std::vector<std::map<std::string, std::string>> bc_list_;
    std::unordered_map<std::string, std::string> bc_list_;
};

#endif // BOUNDARY_CONDITION_CHECKER_HPP