#include "ParameterConverter.hpp"

Parameters ParameterConverter::convertForU(const Parameters& parameters) {
    Parameters result{Parameters(parameters)};
    result.nx = parameters.nx - 1;
    return result;
}

Parameters ParameterConverter::convertForV(const Parameters& parameters) {
    Parameters result{Parameters(parameters)};
    result.ny = parameters.ny - 1;
    return result;
}