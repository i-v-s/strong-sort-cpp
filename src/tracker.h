#ifndef TRACKER_H
#define TRACKER_H
#include <memory>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>

#include "types.h"


using json = nlohmann::json;

namespace strongsort {

constexpr int FEATURE_SIZE = 512;
using Feature = Eigen::Vector<float, FEATURE_SIZE>;
using real = float;


class Tracker;

class StrongSort
{
    const float maxDist;
    Tracker *tracker;
public:
    json dumpTracks() const noexcept;
    StrongSort(real maxDist = 0.2,
               real maxIouDistance = 0.7,
               int maxAge=70,
               int nInit = 3,
               int nnBudget = 100
               );
    static std::unique_ptr<StrongSort> fromJson(const json &config);
    ~StrongSort();
    std::vector<TrackedBox> update(const Eigen::Matrix<real, Eigen::Dynamic, 4> &ltwhs,
                                   const Eigen::VectorX<real> &confidences,
                                   const Eigen::VectorXi &classes,
                                   const Eigen::Matrix<real, Eigen::Dynamic, FEATURE_SIZE> &features,
                                   const std::array<int, 2> &imageSize);

    std::vector<TrackedBox> update(const std::vector<DetectedBox> &boxes,
                                   const Eigen::Matrix<real, Eigen::Dynamic, FEATURE_SIZE> &features,
                                   const std::array<int, 2> &imageSize);
    void incrementAges() noexcept;
};


}
#endif // TRACKER_H
