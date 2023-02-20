#include <iostream>
#include <string>
#include <numeric>
#include <iterator>
#include <map>
#include <set>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <rectangular_lsap.h>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include "tracker.h"

using namespace std;
using namespace Eigen;
using namespace cv;

namespace strongsort {


/*********************************************************** KalmanFilter ***********************************************************/


class KalmanFilter
{
public:
    static constexpr int ndim = 4;
    using FullVector = Vector<real, ndim * 2>;
    using FullMatrix = Matrix<real, ndim * 2, ndim * 2>;
    KalmanFilter();
    pair<FullVector, FullMatrix> initiate(const Vector<real, ndim> &measurement) noexcept;
    pair<FullVector, FullMatrix> predict(const FullVector &mean, const FullMatrix &covariance) noexcept;
    std::pair<Vector<real, ndim>, Matrix<real, ndim, ndim> > project(const FullVector &mean, const FullMatrix &covariance, real confidence = 0.0) const noexcept;
    pair<FullVector, FullMatrix> update(const FullVector &mean, const FullMatrix &covariance,
                                             const Vector<real, ndim> &measurement, real confidence = 0.0);
    VectorX<real> gatingDistance(const FullVector &mean, const FullMatrix &covariance,
                                         const Matrix<real, Eigen::Dynamic, ndim> &measurements,
                                         bool onlyPosition = false) const;
private:
    FullMatrix motionMat = FullMatrix::Identity();
    Matrix<real, ndim, 2 * ndim> updateMat = Matrix<real, ndim, 2 * ndim>::Identity();
    real stdWeightPosition = 1. / 20;
    real stdWeightVelocity = 1. / 160;
};


KalmanFilter::KalmanFilter()
{
    for (int i = 0; i < ndim; ++i)
        motionMat(i, ndim + i) = 1.0;
}

pair<KalmanFilter::FullVector, KalmanFilter::FullMatrix> KalmanFilter::initiate(const Vector<real, ndim> &measurement) noexcept
{
    FullVector mean;
    mean << measurement, Vector<real, ndim>::Zero();
    Vector<real, 8> std;
    std << 2 * stdWeightPosition * measurement[0],   // the center point x
           2 * stdWeightPosition * measurement[1],   // the center point y
           1 * measurement[2],                               // the ratio of width/height
           2 * stdWeightPosition * measurement[3],   // the height
           10 * stdWeightVelocity * measurement[0],
           10 * stdWeightVelocity * measurement[1],
           0.1 * measurement[2],
           10 * stdWeightVelocity * measurement[3];
    auto covariance = std.array().square().matrix().asDiagonal();
    return make_pair(mean, covariance);
}

pair<KalmanFilter::FullVector, KalmanFilter::FullMatrix> KalmanFilter::predict(const FullVector &mean, const FullMatrix &covariance) noexcept
{
    Array<real, ndim * 2, 1> std;
    std << stdWeightPosition * mean[0],
           stdWeightPosition * mean[1],
           1 * mean[2],
           stdWeightPosition * mean[3],
           stdWeightVelocity * mean[0],
           stdWeightVelocity * mean[1],
           0.1 * mean[2],
           stdWeightVelocity * mean[3];

    auto motionCov = std.square().matrix().asDiagonal();
    FullVector nextMean = motionMat * mean;
    FullMatrix nextCovariance = motionMat * covariance * motionMat.transpose() + FullMatrix(motionCov);
    return make_pair(nextMean, nextCovariance);
}

std::pair<Vector<real, KalmanFilter::ndim>, Matrix<real, KalmanFilter::ndim, KalmanFilter::ndim>> KalmanFilter::project(const FullVector &mean, const FullMatrix &covariance, real confidence) const noexcept
{
    Array<real, ndim, 1> std;
    std << stdWeightPosition * mean[3],
           stdWeightPosition * mean[3],
           1e-1,
           stdWeightPosition * mean[3];
    std = (1 - confidence) * std;
    auto innovationCov = std.square().matrix().asDiagonal();
    return make_pair(updateMat * mean, updateMat * covariance * updateMat.transpose() + Matrix<real, ndim, ndim>(innovationCov));
}

std::pair<KalmanFilter::FullVector, KalmanFilter::FullMatrix> KalmanFilter::update(const FullVector &mean, const FullMatrix &covariance, const Eigen::Vector<real, ndim> &measurement, real confidence)
{
    auto [projectedMean, projectedCov] = project(mean, covariance, confidence);

    Matrix<real, 2 * ndim, ndim> kalmanGain = projectedCov.ldlt().solve(updateMat * covariance.transpose()).transpose();

    auto innovation = measurement - projectedMean;
    FullVector newMean = mean + kalmanGain * innovation;
    FullMatrix newCovariance = covariance - kalmanGain * projectedCov * kalmanGain.transpose();
    return make_pair(newMean, newCovariance);
}

Eigen::VectorX<real> KalmanFilter::gatingDistance(const FullVector &mean, const FullMatrix &covariance, const Matrix<real, Dynamic, ndim> &measurements, bool onlyPosition) const
{
    auto [pMean, pCovariance] = project(mean, covariance);

    assert(!onlyPosition);
    /*if (onlyPosition)
    {
        mean, covariance = mean[:2], covariance[:2, :2];
        measurements = measurements[:, :2];
    }*/
    auto d = measurements.transpose().colwise() - pMean;
    VectorX<real> squared_maha = pCovariance.llt().matrixL().solve(d).array().square().colwise().sum();
    return squared_maha;
}



/*********************************************************** NearestNeighborDistanceMetric ***********************************************************/

class NearestNeighborDistanceMetric
{
public:
    enum class Type {
        euclidean,
        cosine
    } const type;
    const float matchingThreshold;

    NearestNeighborDistanceMetric(Type metric, float matchingThreshold, int budget = -1)
        : type(metric), matchingThreshold(matchingThreshold), budget(budget)
    {}

    void partialFit(const Matrix<real, Dynamic, FEATURE_SIZE> &features, const vector<int> &targets, const vector<int> &activeTargets)
    {
        assert(features.rows() == targets.size());
        for (size_t i = 0; i < targets.size(); ++i) {
            auto feature = features.row(i);
            auto target = targets[i];
            auto item = samples.find(target);
            if (item == samples.end())
                samples.emplace(target, vector<Feature> {feature});
            else
            {
                vector<Feature> &features = item->second;
                if (budget > 0 && features.size() >= budget)
                {
                    shift_left(features.begin(), features.end(), 1);
                    features.back() = feature;
                }
                else
                    features.push_back(feature);
            }
        }
        decltype (samples) newSamples;
        for (auto t: activeTargets)
            newSamples.insert(make_pair(t, samples[t]));
        samples = move(newSamples);
    }

    Matrix<real, Dynamic, Dynamic> distance(const Matrix<real, Dynamic, FEATURE_SIZE> &features, const vector<int> &targets)
    {
        /*Compute distance between features and targets.
        Parameters
        ----------
        features : ndarray
            An NxM matrix of N features of dimensionality M.
        targets : List[int]
            A list of targets to match the given `features` against.
        Returns
        -------
        ndarray
            Returns a cost matrix of shape len(targets), len(features), where
            element (i, j) contains the closest squared distance between
            `targets[i]` and `features[j]`.
        */
        Matrix<real, Dynamic, Dynamic> costMatrix = Matrix<real, Dynamic, Dynamic>::Zero(targets.size(), features.rows());
        Matrix<real, Dynamic, FEATURE_SIZE> sampleMatrix;

        for(size_t i = 0; i < targets.size(); ++i)
        {
            const auto &sample = samples[targets[i]];
            sampleMatrix.resize(sample.size(), FEATURE_SIZE);
            copy(sample.begin(), sample.end(), sampleMatrix.rowwise().begin());
            costMatrix.row(i) = metric(sampleMatrix, features);
        }
        return costMatrix;
    }
private:
    static Matrix<real, Dynamic, FEATURE_SIZE> normalized(const Matrix<real, Dynamic, FEATURE_SIZE> &x, float eps = 1e-12) noexcept
    {
        auto norm = x.array().square().rowwise().sum().sqrt().max(eps);
        return x.array().colwise() / norm;
    }
    Matrix<real, Dynamic, Dynamic> metric(const Matrix<real, Dynamic, FEATURE_SIZE> &x, const Matrix<real, Dynamic, FEATURE_SIZE> &y)
    {
        Matrix<real, Dynamic, Dynamic> dist;
        switch(type)
        {
        case Type::cosine:
        {
            auto xn = normalized(x);
            auto yn = normalized(y);
            dist = 1.0 - (xn * yn.transpose()).array();
            break;
        }
        case Type::euclidean:
            assert(!"Not implemented");
            break;
        }
        return dist.colwise().minCoeff();
    }
    int budget;
    map<int, std::vector<Feature>> samples;
};


/*********************************************************** Detection ***********************************************************/

struct Detection
{
    Vector4<real> tlwh;
    size_t index;
    float confidence;
    Feature feature;

    Vector4<real> xyah() const noexcept
    {
        /*Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.*/
        Vector4<real> ret = tlwh;
        ret.block<2, 1>(0, 0) += ret.block<2, 1>(2, 0) / 2;
        ret[2] /= ret[3];
        return ret;
    }
};

/*********************************************************** Track ***********************************************************/

struct Track
{
    enum class State
    {
        Tentative = 1,
        Confirmed = 2,
        Deleted = 3
    };
    uint trackId, classId, lastDetectionIdx;
    int hits = 1, age = 1;
    int timeSinceUpdate = 0;
    float emaAlpha;
    State state = State::Tentative;
    float conf;
    int nInit, maxAge;
    Feature feature;
    KalmanFilter kf;
    KalmanFilter::FullVector mean;
    KalmanFilter::FullMatrix covariance;

    Track(const Eigen::Vector4<real> &detection, int trackId, int classId, float conf, int nInit, int maxAge, float emaAlpha,
          const Feature &feature) :
        trackId(trackId), classId(classId), emaAlpha(emaAlpha), conf(conf), nInit(nInit), maxAge(maxAge), feature(feature / feature.norm())
    {
        tie(mean, covariance) = kf.initiate(detection);
    }

    void incrementAge() noexcept
    {
        age++;
        timeSinceUpdate++;
    }

    void markMissed() noexcept
    {
        if (state == State::Tentative || timeSinceUpdate > maxAge)
            state = State::Deleted;
    }

    void predict() noexcept
    {
        tie(mean, covariance) = kf.predict(mean, covariance);
        age++;
        timeSinceUpdate++;
    }

    TrackedBox result(int width, int height) const noexcept
    {
        Vector4<real> r = tlwh();
        r.block<2, 1>(2, 0) += r.block<2, 1>(0, 0);
        return TrackedBox{clip(r[0] / width), clip(r[1] / height), clip(r[2] / width), clip(r[3] / height), trackId, classId, lastDetectionIdx, conf, timeSinceUpdate};
    }
    TrackedBox result(const cv::Size &imageSize) const noexcept
    {
        return result(imageSize.width, imageSize.height);
    }

    Vector4<real> tlwh() const noexcept
    {
        /*Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        */
        Vector4<real> ret = mean.block<4, 1>(0, 0);
        ret[2] *= ret[3];
        ret.block<2, 1>(0, 0) -= ret.block<2, 1>(2, 0) / 2;
        return ret;
    }

    void update(const Detection &detection, int classId, float conf)
    {
        /*Perform Kalman filter measurement update step and update the feature
        cache.
        Parameters
        ----------
        detection : Detection
            The associated detection.
        */
        this->conf = conf;
        this->classId = classId;
        this->lastDetectionIdx = detection.index;
        tie(mean, covariance) = kf.update(mean, covariance, detection.xyah(), detection.confidence);

        Feature detectionFeature = detection.feature / detection.feature.norm();
        Feature smoothFeature = emaAlpha * feature + (1 - emaAlpha) * detectionFeature;
        feature = smoothFeature / smoothFeature.norm();

        hits++;
        timeSinceUpdate = 0;
        if (state == State::Tentative && hits >= nInit)
            state = State::Confirmed;
    }
};

/*********************************************************** Linear assignment ***********************************************************/

constexpr float INFTY_COST = 1e+5;

pair<vector<int64_t>, vector<int64_t>> linearSumAssignment(const MatrixX<real> &costMatrix)
{
    int maximize = 0;

    auto numRows = costMatrix.rows();
    auto numCols = costMatrix.cols();
    auto dim = min(numRows, numCols);
    vector<int64_t> a(dim), b(dim);
    Matrix<double, Dynamic, Dynamic, RowMajor> cm = costMatrix.cast<double>();

    int ret = solve_rectangular_linear_sum_assignment(
      numRows, numCols, cm.data(), false, a.data(), b.data());
    if (ret == RECTANGULAR_LSAP_INFEASIBLE)
        throw runtime_error("cost matrix is infeasible");
    else if (ret == RECTANGULAR_LSAP_INVALID)
        throw runtime_error("matrix contains invalid numeric entries");
    return make_pair(a, b);
}


template<typename DistanceMetric>
tuple<vector<pair<int, int>>, vector<int>, vector<int>> minCostMatching(
            DistanceMetric distanceMetric,
            float maxDistance,
            const vector<Track> &tracks,
            const vector<Detection> &detections,
            const vector<int> &trackIndices,
            const vector<int> &detectionIndices)
{
    /*Solve linear assignment problem.
    Parameters
    ----------
    distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
        The distance metric is given a list of tracks and detections as well as
        a list of N track indices and M detection indices. The metric should
        return the NxM dimensional cost matrix, where element (i, j) is the
        association cost between the i-th track in the given track indices and
        the j-th detection in the given detection_indices.
    max_distance : float
        Gating threshold. Associations with cost larger than this value are
        disregarded.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : List[int]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above).
    detection_indices : List[int]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above).
    Returns
    -------
    (List[(int, int)], List[int], List[int])
        Returns a tuple with the following three entries:
        * A list of matched track and detection indices.
        * A list of unmatched track indices.
        * A list of unmatched detection indices.
    */

    if (detectionIndices.empty() || trackIndices.empty())
        return make_tuple(vector<pair<int, int>>(), trackIndices, detectionIndices); // Nothing to match.

    auto costMatrix = distanceMetric(tracks, detections, trackIndices, detectionIndices);
    for(auto &value : costMatrix.reshaped())
        if(value > maxDistance)
            value = maxDistance + 1e-5;

    auto [rowIndices, colIndices] = linearSumAssignment(costMatrix);

    vector<pair<int, int>> matches;
    vector<int> unmatchedTracks, unmatchedDetections;

    assert(rowIndices.size() == colIndices.size());

    for (size_t col = 0; col < detectionIndices.size(); ++col)
        if (find(colIndices.begin(), colIndices.end(), col) == colIndices.end())
            unmatchedDetections.push_back(detectionIndices[col]);
    for (size_t row = 0; row < trackIndices.size(); ++row)
        if (find(rowIndices.begin(), rowIndices.end(), row) == rowIndices.end())
            unmatchedTracks.push_back(trackIndices[row]);
    for (auto row = rowIndices.begin(), col = colIndices.begin(); row != rowIndices.end() && col != colIndices.end(); ++row, ++col)
    {
        auto trackIdx = trackIndices[*row];
        auto detectionIdx = detectionIndices[*col];
        if (costMatrix(*row, *col) > maxDistance)
        {
            unmatchedTracks.push_back(trackIdx);
            unmatchedDetections.push_back(detectionIdx);
        }
        else
            matches.push_back(make_pair(trackIdx, detectionIdx));
    }
    return make_tuple(matches, unmatchedTracks, unmatchedDetections);
}

/*********************************************************** IOU ***********************************************************/

VectorX<real> iou(const Array4<real> &tlwh, const Array<real, Dynamic, 4> &candidates)
{
    /*Compute intersection over union.

    Parameters
    ----------
    bbox : ndarray
        A bounding box in format `(top left x, top left y, width, height)`.
    candidates : ndarray
        A matrix of candidate bounding boxes (one per row) in the same format
        as `tlwh`.

    Returns
    -------
    ndarray
        The intersection over union in [0, 1] between the `bbox` and each
        candidate. A higher score means a larger fraction of the `bbox` is
        occluded by the candidate.
    */
    auto rows = candidates.rows();
    Array2<real> boxTL = tlwh.block<2, 1>(0, 0), boxBR = boxTL + tlwh.block<2, 1>(2, 0);
    Array<real, Dynamic, 4> tlbr(rows, 4);
    for (size_t i = 0; i < rows; ++i)
    {
        Array4<real> candidate = candidates.row(i);
        auto candidateTL = candidate.block<2, 1>(0, 0);
        tlbr.block<1, 2>(i, 0) = candidateTL.max(boxTL);
        tlbr.block<1, 2>(i, 2) = (candidateTL + candidate.block<2, 1>(2, 0)).min(boxBR);
    }
    auto wh = (tlbr.block(0, 2, rows, 2) - tlbr.block(0, 0, rows, 2)).max(0.0);
    auto areaIntersection = wh.col(0) * wh.col(1);
    auto areaBox = tlwh[2] * tlwh[3];
    auto areaCandidates = candidates.col(2) * candidates.col(3);
    return areaIntersection / (areaBox + areaCandidates - areaIntersection);
}

MatrixX<real> iouCost(
      const vector<Track> &tracks,
      const vector<Detection> &detections,
      const vector<int> &trackIndices,
      const vector<int> &detectionIndices)
 {
    /*An intersection over union distance metric.

    Parameters
    ----------
    tracks : List[deep_sort.track.Track]
        A list of tracks.
    detections : List[deep_sort.detection.Detection]
        A list of detections.
    track_indices : Optional[List[int]]
        A list of indices to tracks that should be matched. Defaults to
        all `tracks`.
    detection_indices : Optional[List[int]]
        A list of indices to detections that should be matched. Defaults
        to all `detections`.

    Returns
    -------
    ndarray
        Returns a cost matrix of shape
        len(track_indices), len(detection_indices) where entry (i, j) is
        `1 - iou(tracks[track_indices[i]], detections[detection_indices[j]])`.
    */
    MatrixX<real> costMatrix = MatrixX<real>::Zero(trackIndices.size(), detectionIndices.size());
    for (size_t row = 0; row < trackIndices.size(); ++row)
    {
        int trackIdx = trackIndices[row];
        if (tracks[trackIdx].timeSinceUpdate > 1)
        {
            costMatrix.row(row).fill(INFTY_COST);
            continue;
        }

        auto bbox = tracks[trackIdx].tlwh();
        Matrix<real, Dynamic, 4> candidates(detectionIndices.size(), 4);
        for (size_t i = 0; i < detectionIndices.size(); ++i)
            candidates.row(i) = detections[detectionIndices[i]].tlwh;
        costMatrix.row(row) = 1.0 - iou(bbox, candidates).array();
    }
    return costMatrix;
}

template<typename DistanceMetric>
tuple<vector<pair<int, int>>, vector<int>, vector<int>> matchingCascade(
        DistanceMetric distanceMetric,
        float maxDistance,
        const vector<Track> &tracks,
        const vector<Detection> &detections,
        const vector<int> &trackIndices,
        const vector<int> &detectionIndices)
{
    /*Run matching cascade.
    Parameters
    ----------
    distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
        The distance metric is given a list of tracks and detections as well as
        a list of N track indices and M detection indices. The metric should
        return the NxM dimensional cost matrix, where element (i, j) is the
        association cost between the i-th track in the given track indices and
        the j-th detection in the given detection indices.
    max_distance : float
        Gating threshold. Associations with cost larger than this value are
        disregarded.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : Optional[List[int]]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above). Defaults to all tracks.
    detection_indices : Optional[List[int]]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above). Defaults to all
        detections.
    Returns
    -------
    (List[(int, int)], List[int], List[int])
        Returns a tuple with the following three entries:
        * A list of matched track and detection indices.
        * A list of unmatched track indices.
        * A list of unmatched detection indices.
    */

    auto unmatchedDetections = detectionIndices;
    auto match = minCostMatching(distanceMetric, maxDistance, tracks, detections, trackIndices, unmatchedDetections);
    auto matches = get<0>(match);
    unmatchedDetections = get<2>(match);
    vector<int> unmatchedTracks;
    set<int> mf;
    transform(matches.begin(), matches.end(), inserter(mf, mf.begin()), [] (auto &p) { return p.first; });
    set_difference(trackIndices.begin(), trackIndices.end(),
                   mf.begin(), mf.end(),
                   inserter(unmatchedTracks, unmatchedTracks.end()));
    return make_tuple(matches, unmatchedTracks, unmatchedDetections);
}

const float chi2inv95[] = {3.8415, 5.9915, 7.8147, 9.4877, 11.070, 12.592, 14.067, 15.507, 16.919};

void gateCostMatrix(
        Matrix<real, Dynamic, Dynamic> &costMatrix,
        const vector<Track> &tracks,
        const vector<Detection> &detections,
        const vector<int> trackIndices,
        const vector<int> detectionIndices,
        float gatedCost = 100000.0f,
        bool onlyPosition = false)
{
    /*Invalidate infeasible entries in cost matrix based on the state
    distributions obtained by Kalman filtering.
    Parameters
    ----------
    kf : The Kalman filter.
    cost_matrix : ndarray
        The NxM dimensional cost matrix, where N is the number of track indices
        and M is the number of detection indices, such that entry (i, j) is the
        association cost between `tracks[track_indices[i]]` and
        `detections[detection_indices[j]]`.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : List[int]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above).
    detection_indices : List[int]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above).
    gated_cost : Optional[float]
        Entries in the cost matrix corresponding to infeasible associations are
        set this value. Defaults to a very large value.
    only_position : Optional[bool]
        If True, only the x, y position of the state distribution is considered
        during gating. Defaults to False.
    Returns
    -------
    ndarray
        Returns the modified cost matrix.
    */
    size_t gatingDim = onlyPosition ? 2 : 4;
    auto gatingThreshold = chi2inv95[gatingDim - 1];
    Matrix<real, Dynamic, KalmanFilter::ndim> measurements(detectionIndices.size(), KalmanFilter::ndim);
    transform(detectionIndices.begin(), detectionIndices.end(), measurements.rowwise().begin(), [&detections] (auto i) { return detections[i].xyah(); });

    for (size_t row = 0; row < trackIndices.size(); ++row)
    {
        auto trackIdx = trackIndices[row];
        auto &track = tracks[trackIdx];
        auto gatingDistance = track.kf.gatingDistance(track.mean, track.covariance, measurements, onlyPosition);
        for (size_t col = 0; col < costMatrix.cols(); ++col)
        {
            auto value = (gatingDistance[col] > gatingThreshold) ? gatedCost : costMatrix(row, col);
            costMatrix(row, col) = 0.995 * value + (1 - 0.995) * gatingDistance[col];
        }
    }
}

/*********************************************************** Tracker ***********************************************************/


class Tracker
{
    int nextId = 1;
    NearestNeighborDistanceMetric metric;
    const real maxIouDistance;
    const int maxAge, nInit;
    const real lambda, emaAlpha, mcLambda;
public:
    std::vector<Track> tracks;

    Tracker(NearestNeighborDistanceMetric && metric,
            real maxIouDistance = 0.9,
            int maxAge = 30,
            int nInit = 3,
            real lambda = 0,
            real emaAlpha = 0.9,
            real mcLambda = 0.995) noexcept :
        metric(metric), maxIouDistance(maxIouDistance), maxAge(maxAge), nInit(nInit), lambda(lambda), emaAlpha(emaAlpha), mcLambda(mcLambda)
    {
    }

    void predict() noexcept
    {
        for(auto &track: tracks)
            track.predict();
    }

    void incrementAges() noexcept
    {
        for (auto &track: tracks)
        {
            track.incrementAge();
            track.markMissed();
        }
    }

    json dumpTracks() const noexcept
    {
        json result = json::array();
        for (const auto &track: tracks)
        {
            json item;
            item["age"] = track.age;
            item["conf"] = track.conf;
            item["hits"] = track.hits;
            vector<real> mean(track.mean.begin(), track.mean.end());
            item["mean"] = mean;
            item["state"] = track.state;
            item["time_since_update"] = track.timeSinceUpdate;
            item["track_id"] = track.trackId;
            result.push_back(item);
        }
        return result;
    }

    void update(const vector<Detection> &detections, const VectorX<int> &classes, const VectorX<real> &confidences)
    {
        // Run matching cascade.
        auto [matches, unmatchedTracks, unmatchedDetections] = match(detections);

        // Update track set.
        for (auto [trackIdx, detectionIdx]: matches)
            tracks[trackIdx].update(detections[detectionIdx], classes[detectionIdx], confidences[detectionIdx]);

        for (auto trackIdx: unmatchedTracks)
            tracks[trackIdx].markMissed();
        for (auto detectionIdx: unmatchedDetections)
            initiateTrack(detections[detectionIdx], classes[detectionIdx], confidences[detectionIdx]);
        erase_if(tracks, [] (const auto &track) { return track.state == Track::State::Deleted; });

        // Update distance metric.
        vector<int> activeTargets, targets;
        for (const auto &track: tracks)
            if (track.state == Track::State::Confirmed) {
                activeTargets.push_back(track.trackId);
                targets.push_back(track.trackId);
            }
        Matrix<real, Dynamic, FEATURE_SIZE> features(targets.size(), FEATURE_SIZE);
        auto dst = features.rowwise().begin();
        for (const auto &track: tracks)
            if (track.state == Track::State::Confirmed)
                *(dst++) = track.feature;
        metric.partialFit(features, targets, activeTargets);
    }
private:
    tuple<vector<pair<int, int>>, vector<int>, vector<int>> match(const vector<Detection> &detections)
    {
        // Split track set into confirmed and unconfirmed tracks.
        vector<int> confirmedTracks, unconfirmedTracks;
        for (size_t i = 0; i < tracks.size(); ++i)
        {
            auto &track = tracks[i];
            if (track.state == Track::State::Confirmed)
                confirmedTracks.push_back(i);
            else
                unconfirmedTracks.push_back(i);
        }

        auto gatedMetric = [this] (const vector<Track> &tracks, const vector<Detection> &dets, const vector<int> &trackIndices, const vector<int> &detectionIndices)
        {
            Matrix<real, Dynamic, FEATURE_SIZE> features(detectionIndices.size(), FEATURE_SIZE);
            transform(detectionIndices.begin(), detectionIndices.end(), features.rowwise().begin(), [&dets] (int i) { return dets[i].feature; });
            vector<int> targets(trackIndices.size());
            transform(trackIndices.begin(), trackIndices.end(), targets.begin(), [&tracks] (int i) { return tracks[i].trackId;});
            auto costMatrix = metric.distance(features, targets);
            gateCostMatrix(costMatrix, tracks, dets, trackIndices, detectionIndices);
            return costMatrix;
        };

        // Associate confirmed tracks using appearance features.
        vector<int> detectionIndices(detections.size());
        iota(detectionIndices.begin(), detectionIndices.end(), 0);
        auto [matchesA, unmatchedTracksA, unmatchedDetections] =
                matchingCascade(gatedMetric, metric.matchingThreshold, tracks, detections, confirmedTracks, detectionIndices);

        // Associate remaining tracks together with unconfirmed tracks using IOU.
        vector<int> iouTrackCandidates = unconfirmedTracks;
        copy_if(unmatchedTracksA.begin(), unmatchedTracksA.end(),
                back_inserter(iouTrackCandidates), [this] (int i) { return tracks[i].timeSinceUpdate == 1; });

        erase_if(unmatchedTracksA, [this] (int i) { return tracks[i].timeSinceUpdate == 1; });

        vector<pair<int, int>> matchesB;
        vector<int> unmatchedTracksB;
        tie(matchesB, unmatchedTracksB, unmatchedDetections) = minCostMatching(
                iouCost, maxIouDistance, tracks,
                detections, iouTrackCandidates, unmatchedDetections);

        auto matches = move(matchesA);
        move(matchesB.begin(), matchesB.end(), back_inserter(matches));

        vector<int> unmatchedTracks;
        set_union(unmatchedTracksA.begin(), unmatchedTracksA.end(),
                  unmatchedTracksB.begin(), unmatchedTracksB.end(),
                  back_inserter(unmatchedTracks));
        return make_tuple(matches, unmatchedTracks, unmatchedDetections);
    }

    void initiateTrack(const Detection &detection, int classId, float conf) noexcept
    {
        tracks.emplace_back(detection.xyah(), nextId, classId, conf, nInit, maxAge, emaAlpha, detection.feature);
        nextId++;
    }
};



/*********************************************************** StrongSORT ***********************************************************/

json StrongSort::dumpTracks() const noexcept
{
    return tracker->dumpTracks();
}

unordered_set<uint> StrongSort::trackIds() const noexcept
{
    unordered_set<uint> result(tracker->tracks.size());
    for (const Track &t: tracker->tracks)
        result.emplace(t.trackId);
    return result;
}

StrongSort::StrongSort(real maxDist, real maxIouDistance, int maxAge, int nInit, int nnBudget) :
    maxDist(maxDist)
{
    auto &&metric = NearestNeighborDistanceMetric(NearestNeighborDistanceMetric::Type::cosine, maxDist, nnBudget);
    tracker = new Tracker(move(metric), maxIouDistance, maxAge, nInit);
}

std::unique_ptr<StrongSort> StrongSort::fromJson(const json &config)
{
    return make_unique<StrongSort>(
            config.contains("max_dist") ? config["max_dist"].get<real>() : 0.2,
            config.contains("max_iou_distance") ? config["max_iou_distance"].get<real>() : 0.7
        );
}

StrongSort::~StrongSort()
{
    delete tracker;
}

vector<TrackedBox> StrongSort::update(const Matrix<real, Dynamic, 4> &ltwhs,
                                      const VectorX<real> &confidences,
                                      const VectorX<int> &classes,
                                      const Matrix<real, Dynamic, FEATURE_SIZE> &features,
                                      const array<int, 2> &imageSize)
{
    auto [w, h] = imageSize;
    // generate detections
    vector<Detection> detections;
    detections.reserve(ltwhs.rows());
    for (size_t i = 0; i < ltwhs.rows(); ++i)
    {
        auto ltwh = ltwhs.block<1, 4>(i, 0);
        detections.emplace_back(ltwh, i, confidences[i], features.row(i));
    }

    // update tracker
    tracker->predict();
    tracker->update(detections, classes, confidences);

    // output bbox identities
    vector<TrackedBox> outputs;
    for (const auto &track: tracker->tracks)
        if (track.state == Track::State::Confirmed && track.timeSinceUpdate <= 1) {
            auto tb = track.result(w, h);
            if (!tb.empty())
                outputs.push_back(tb);
        }
    return outputs;
}

vector<TrackedBox> StrongSort::update(const vector<DetectedBox> &boxes, const Matrix<real, Dynamic, FEATURE_SIZE> &features, const array<int, 2> &imageSize)
{
    size_t size = boxes.size();
    Matrix<real, Dynamic, 4> ltwhs(size, 4);
    VectorX<real> confidences(size);
    VectorX<int> classes(size);
    const auto [w, h] = imageSize;
    for (size_t i = 0; i < size; ++i)
    {
        const DetectedBox &box = boxes[i];
        Vector4<real> ltwh(box.x1, box.y1, box.x2, box.y2);
        ltwh = ltwh.array().min(1.0).max(0.0);
        ltwh.block<2, 1>(2, 0) -= ltwh.block<2, 1>(0, 0);
        ltwh = ltwh.array() * Array4<real>(w, h, w, h);
        ltwhs.row(i) = ltwh;
        confidences[i] = box.confidence;
        classes[i] = static_cast<int>(box.classId);
    }
    return update(ltwhs, confidences, classes, features, imageSize);
}

void StrongSort::incrementAges() noexcept
{
    tracker->incrementAges();
}

}
