#include "BoundaryConditionChecker.hpp"

BoundaryConditionChecker::BoundaryConditionChecker(const std::string& d_or_n_or_t) 
    : bc_list_()
    {
    if (d_or_n_or_t == "D") {
        bc_list_ = std::unordered_map<std::string, std::string>{
            {"t", "D"},
            {"b", "D"},
            {"l", "D"},
            {"r", "D"}
        };
    } else if (d_or_n_or_t == "N") {
        bc_list_ = std::unordered_map<std::string, std::string>{
            {"t", "N"},
            {"b", "N"},
            {"l", "N"},
            {"r", "N"}
        };
    } else if (d_or_n_or_t == "T") {
        bc_list_ = std::unordered_map<std::string, std::string>{
            {"t", "T"},
            {"b", "T"},
            {"l", "T"},
            {"r", "T"}
        };
    } else {
        throw std::invalid_argument("\nString must be D or N!!!");
    }
}

bool BoundaryConditionChecker::checkCondition(
    const std::string& side,
    const std::string& d_or_n_or_t
) const {
    auto iter = bc_list_.find(side);
    assert(iter != bc_list_.end());
    return iter->second == d_or_n_or_t;
}

bool BoundaryConditionChecker::isPureDirichlet() const {
    std::vector<std::string> sides{"t", "b", "l", "r"};
    for (auto& side : sides) {
        if (!this->isDirichlet(side)) return false;
    }
    return true;
}

bool BoundaryConditionChecker::isPureNeumann() const {
    std::vector<std::string> sides{"t", "b", "l", "r"};
    for (auto& side : sides) {
        if (!this->isNeumann(side)) return false;
    }
    return true;
}

bool BoundaryConditionChecker::isPurePeriodic() const {
    std::vector<std::string> sides{"t", "b", "l", "r"};
    for (auto& side : sides) {
        if (!this->isPeriodic(side)) return false;
    }
    return true;
}

bool BoundaryConditionChecker::isMixedDirichletNeumann() const {
    std::vector<std::string> sides{"t", "b", "l", "r"};
    for (auto& side : sides) {
        if (!(this->isDirichlet(side) || this->isNeumann(side))) return false;
    }
    return true;
}

std::string BoundaryConditionChecker::describe() const {
    auto code = [this](const std::string& s){
        if (this->isDirichlet(s)) return 'D';
        if (this->isNeumann(s))   return 'N';
        if (this->isPeriodic(s))  return 'T';
        return '?';
    };
    std::string out;
    out += code("t"); out += code("b"); out += code("l"); out += code("r");
    return out;
}