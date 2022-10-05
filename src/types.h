#ifndef TYPES_H
#define TYPES_H
#include <opencv2/core.hpp>

struct RelativeBox
{
    cv::Rect rect(const cv::Size &imageSize) const noexcept;
    cv::Rect rect(const cv::Mat &image) const noexcept;
    float area() const noexcept;
    float width() const noexcept;
    float height() const noexcept;
    std::array<float, 4> array() const noexcept;
    std::pair<float, float> iou(const RelativeBox &b) const noexcept;

    float x1, y1, x2, y2;
};

struct TrackedBox: public RelativeBox
{
    uint trackId, classId;
    float confidence;
};

struct DetectedBox: public RelativeBox
{
    DetectedBox(const float *src, float w, float h) noexcept;
    bool operator ==(const DetectedBox &other) const noexcept;

    uint classId, batchId;
    float confidence;
};

#endif // TYPES_H
