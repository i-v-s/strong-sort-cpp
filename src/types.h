#ifndef TYPES_H
#define TYPES_H
#include <opencv2/core.hpp>

struct RelativeBox
{
    cv::Rect rect(const cv::Size &imageSize) const noexcept;
    cv::Rect rect(const cv::Mat &image) const noexcept;
    float area() const noexcept;
    bool empty() const noexcept;
    float width() const noexcept;
    float height() const noexcept;
    std::array<float, 4> array() const noexcept;
    std::pair<float, float> iou(const RelativeBox &b) const noexcept;

    float x1 = 0.f, y1 = 0.f, x2 = 0.f, y2 = 0.f;
};

struct TrackedBox: public RelativeBox
{
    uint trackId, classId, detectionId;
    float confidence;
    int timeSinceUpdate;
};

struct DetectedBox: public RelativeBox
{
    DetectedBox(const float *src, float w, float h) noexcept;
    DetectedBox(uint classId, uint batchId, float confidence, const std::array<float, 4> &box, const std::array<uint, 2> &size) noexcept;
    bool operator ==(const DetectedBox &other) const noexcept;

    uint classId, batchId;
    float confidence;
};

template<typename T>
float clip(T value, T min_ = 0, T max_ = 1) noexcept
{
    return std::min(std::max(value, min_), max_);
}

#endif // TYPES_H
