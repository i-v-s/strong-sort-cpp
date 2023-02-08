#include <algorithm>
#include <exception>

using namespace std;

#include "types.h"

DetectedBox::DetectedBox(const float *src, float w, float h) noexcept :
    RelativeBox{clip(src[1] / w), clip(src[2] / h), clip(src[3] / w), clip(src[4] / h)},
    batchId(src[0]), classId(static_cast<uint>(src[5])),
    confidence(src[6])
{}

DetectedBox::DetectedBox(uint classId, uint batchId, float confidence, const std::array<float, 4> &box, const std::array<uint, 2> &size) noexcept :
    RelativeBox{clip(box[0] / size[0]), clip(box[1] / size[1]), clip(box[2] / size[0]), clip(box[3] / size[1])},
    batchId(batchId), classId(classId),
    confidence(confidence)
{}

bool DetectedBox::operator ==(const DetectedBox &other) const noexcept
{
    return classId == other.classId && batchId == other.batchId && confidence == other.confidence
            && x1 == other.x1 && y1 == other.y1 && x2 == other.x2 && y2 == other.y2;
}

cv::Rect RelativeBox::rect(const cv::Size &imageSize) const noexcept
{
    int w = imageSize.width, h = imageSize.height;
    return cv::Rect(x1 * w, y1 * h, (x2 - x1) * w, (y2 - y1) * h);
}

cv::Rect RelativeBox::rect(const cv::Mat &image) const noexcept
{
    return rect(cv::Size(image.cols, image.rows));
}

float RelativeBox::area() const noexcept
{
    return (x2 - x1) * (y2 - y1);
}

bool RelativeBox::empty() const noexcept
{
    return x2 <= x1 || y2 <= y1;
}

float RelativeBox::width() const noexcept
{
    return x2 - x1;
}

float RelativeBox::height() const noexcept
{
    return y2 - y1;
}

array<float, 4> RelativeBox::array() const noexcept
{
    return {x1, y1, x2, y2};
}

pair<float, float> RelativeBox::iou(const RelativeBox &b) const noexcept
{
    auto &a = *this;
    assert(a.x1 < a.x2);
    assert(a.y1 < a.y2);
    assert(b.x1 < b.x2);
    assert(b.y1 < b.y2);

    // Determine the coordinates of the intersection rectangle
    float xLeft = max(a.x1, b.x1),
          yTop = max(a.y1, b.y1),
          xRight = min(a.x2, b.x2),
          yBottom = min(a.y2, b.y2);

    if (xRight < xLeft || yBottom < yTop)
        return make_pair(0.0f, 0.0f);

    // The intersection of two axis-aligned bounding boxes is always an
    // axis-aligned bounding box
    float intersectionArea = (xRight - xLeft) * (yBottom - yTop);

    // Compute the area of both AABBs
    float area1 = a.area();
    float area2 = b.area();

    float inter1 = intersectionArea / area1;
    float inter2 = intersectionArea / area2;

    float maxInter = max(inter1, inter2);

    // Compute the intersection over union by taking the intersection
    // area and dividing it by the sum of prediction + ground-truth
    // areas - the interesection area
    float iou = intersectionArea / (area1 + area2 - intersectionArea);
    assert(iou >= 0.0);
    assert(iou <= 1.0);

    return make_pair(iou, maxInter);
}
