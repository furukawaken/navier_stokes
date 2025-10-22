#ifndef PARAMETER_CONVERTER_HPP
#define PARAMETER_CONVERTER_HPP

#include "Parameters.hpp"

class ParameterConverter {
    public:
    ~ParameterConverter() {}

    Parameters convertForU(const Parameters& parameters);

    Parameters convertForV(const Parameters& parameters);
};

#endif //PARAMETER_CONVERTER_HPP