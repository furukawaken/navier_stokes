#include "IndexMapping.hpp"

LinearIndexMapping::LinearIndexMapping(int nx, int ny) : nx(nx), ny(ny) {}

int LinearIndexMapping::toIndex(int i, int j) const {
    return i + nx * j;
}

std::pair<int, int> LinearIndexMapping::fromIndex(int l) const {
    int j = l / nx;
    int i = l % nx;
    return {i, j};
}