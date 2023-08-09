#include "../include/fitness.h"
#include <unordered_map>
double Fitness::_corr(VecD y_pred_cur_date, VecD y_cur_date){
    for (int j = 0; j < y_pred_cur_date.size(); j++){
        if (std::isnan(y_pred_cur_date(j)) || std::isnan(y_cur_date(j))){
            y_cur_date(j) = np.nan
            y_pred_cur_date(j) = np.nan
        }
    }

    if (y_pred_cur_date.isNaN().select(1,0).sum() == y_pred_cur_date.size()) continue;
    VecD y_pred_demean = y_pred_cur_date - y_pred_cur_date.mean();
    VecD y_demean = y_cur_date - y_cur_date.mean();
    double top = y_pred_demean.cwiseProduct(y_demean).array().sum().mean();
    double bottom = std::sqrt(y_pred_demean.square().sum() * y_demean.sum());
    double corr = top / bottom;
    return corr;
}

ArrayD Fitness::_rank(ArrayD x){
    ArrayD x_copy = x;
    sort(x_copy.data(), x_copy.data() + x_copy.size());
    unordered_map<double,int> m;
    int i = 0; 
    while (i < x_copy.size()){
        int j = i + 1;
        while (j < x_copy.size() && x_copy(i) == x_copy(j)) j++;
        m[x_copy(i)] = (i + 1 + j) * 2;
    }
    i = 0;
    for (; i < x_copy.size(); i++){
        x_copy(i) = m[x_copy(i)];
    }
    return x_copy;
}

double Fitness::weighted_pearson_corr(MatD y, MatD y_pred, ArrayD weight){
    VecD y_clean = y(weight,Eigen::all), y_pred_clean = y_pred(weight,Eigen::all);
    assert(y_clean.size() == y_pred_clean.size());
    double total_IC = 0.0;
    int n_dates = y.size(), valid_count = 0;
    for (int i = 0; i < n_dates; i++){
        ArrayD y_pred_cur_date = y_pred_clean.row(i), y_cur_date = y_clean.row(i);
        ArrayD y_pred_cur_date_rank = _rank(ArrayD y_pred_cur_date),  y_cur_date_rank = _rank(y_cur_date);
        double corr = _corr(y_pred_cur_date_rank,y_cur_date_rank);
        if (!std::isnan(corr)){
            total_IC += corr;
            valid_count+++;
        }
    }
    if (valid_count > 0){
        total_IC /= valid_count;
    }
    return total_IC;
}

double Fitness::weighted_spearman_corr(MatD y, MatD y_pred, ArrayD weight){
    VecD y_clean = y(weight,Eigen::all), y_pred_clean = y_pred(weight,Eigen::all);
    assert(y_clean.size() == y_pred_clean.size());
    double total_IC = 0.0;
    int n_dates = y.size(), valid_count = 0;
    for (int i = 0; i < n_dates; i++){
        ArrayD y_pred_cur_date = y_pred_clean.row(i), y_cur_date = y_clean.row(i);
        double corr = _corr(y_pred_cur_date,y_cur_date);
        if (!std::isnan(corr)){
            total_IC += corr;
            valid_count+++;
        }
    }
    if (valid_count > 0){
        total_IC /= valid_count;
    }
    return total_IC;
}