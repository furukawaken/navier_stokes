#ifndef INDEX_MAPPING_HPP
#define INDEX_MAPPING_HPP

#include <utility>

class IndexMappingFactory {
    public:
        virtual ~IndexMappingFactory() = default;

        virtual int toIndex(int i, int j) const = 0;
        virtual std::pair<int, int> fromIndex(int l) const = 0;
};

class LinearIndexMapping : public IndexMappingFactory {
    public:
        LinearIndexMapping(int nx, int ny);

        int toIndex(int i, int j) const override;

        std::pair<int, int> fromIndex(int l) const override;

    private:
        int nx;
        int ny;
};

#endif //INDEX_MAPPING_HPP