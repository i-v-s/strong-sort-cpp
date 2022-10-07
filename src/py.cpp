#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "tracker.h"

using namespace strong_sort;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)


namespace py = pybind11;

PYBIND11_MODULE(strong_sort_cpp, m) {
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

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
