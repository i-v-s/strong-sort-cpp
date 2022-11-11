#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "tracker.h"

using namespace strongsort;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)


namespace py = pybind11;

PYBIND11_MODULE(strongsort_py, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: python_example

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";

    py::class_<StrongSort>(m, "StrongSort")
        .def(py::init<float, float, int, int, int>(),
             py::arg("max_dist") = 0.2, py::arg("max_iou_distance") = 0.7, py::arg("max_age") = 70, py::arg("n_init") = 3, py::arg("nn_budget") = 100)
        .def("increment_ages", &StrongSort::incrementAges, "Increment tracks ages, if no detections found")
        .def("update", &StrongSort::update, "Update tracker state with new detections",
             py::arg("ltwhs"), py::arg("confidences"), py::arg("classes"), py::arg("features"), py::arg("image_size"));

    py::class_<TrackedBox>(m, "TrackedBox")
        .def_readwrite("x1", &TrackedBox::x1)
        .def_readwrite("y1", &TrackedBox::y1)
        .def_readwrite("x2", &TrackedBox::x2)
        .def_readwrite("y2", &TrackedBox::y2)
        .def_readwrite("track_id", &TrackedBox::trackId)
        .def_readwrite("class_id", &TrackedBox::classId)
        .def_readwrite("detection_id", &TrackedBox::detectionId)
        .def_readwrite("confidence", &TrackedBox::confidence)
        .def_readwrite("time_since_update", &TrackedBox::timeSinceUpdate)
        .def_property_readonly("xyxy", [] (const TrackedBox &b) { return Eigen::Vector4f(b.x1, b.y1, b.x2, b.y2); });

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
