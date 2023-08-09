#ifndef _FITNESS_H_
#define _FITNESS_H_

#include <Eigen/Dense>

#ifdef EXPORT
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
namespace py=pybind11;
#endif

typedef Eigen::MatrixXd MatD;
typedef Eigen::VectorXd VecD;
typedef Eigen::ArrayXd ArrayD;

class Fitness{
public:

    double weighted_pearson_corr(MatD y, MatD y_pred, ArrayD weight);
    double weighted_spearman_corr(MatD y, MatD y_pred, ArrayD weight);
    double weighted_information_ratio(MatD y, MatD y_pred, ArrayD weight);

private:

    double _corr(VecD y, VecD y_pred);
    VecD _rank(VecD x);
};

#ifdef EXPORT
PYBIND11_MODULE(pybind11_eigen,m){
    m.doc() = "fitness calculation by c++";
    py::class_<Fitness>(m,"fitness")
        .def(py::init())
        .def("weighted_pearson_corr",&Fitness::weighted_pearson_corr)
        .def("weighted_spearman_corr",&Fitness::weighted_spearman_corr)
        .def("weighted_information_ratio",&Fitness::weighted_information_ratio)
}
#endif

#endif