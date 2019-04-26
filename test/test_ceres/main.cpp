
#include "ceres/ceres.h"
#include <iostream>
#include <fstream>
#include "opencv2/tracking/kalman_filters.hpp"
#include "Eigen/Core"
#include <opencv2/core/eigen.hpp>

std::vector<double> g_x_arr;
std::vector<double> g_u_arr;
std::vector<double> g_q_arr;
std::vector<double> g_m_arr;
std::vector<double> g_r_arr;

class UnivariateNonstationaryGrowthModel
{

public:
    template <typename T>
    static void stateConversionFunction(const T& x_k, const T& u_k, const T& v_k, T& x_kplus1)
    {
        x_kplus1 = T(0.5) * x_k + T(25) * (x_k / (x_k * x_k + T(1))) + T(8) * cos(T(1.2) * (u_k - T(1))) + v_k;
    }

    template <typename T>
    static void measurementFunction(const T& x_k, const T& n_k, T& z_k)
    {
        z_k = x_k + n_k; //x*x/20.0 + r;
    }
};

void prepare_data(const int nIterations)
{
    using namespace cv;

    int MP = 1;
    int DP = 1;
    int CP = 0;
    int type = CV_64F;

    Eigen::MatrixXd processNoiseCov = Eigen::MatrixXd::Zero(DP, DP);
    processNoiseCov(0, 0) = 1.0;
    Eigen::MatrixXd processNoiseCovSqrt = processNoiseCov.cwiseSqrt();

    Eigen::MatrixXd measurementNoiseCov = Eigen::MatrixXd::Zero(MP, MP);
    measurementNoiseCov(0, 0) = 1.0;
    Eigen::MatrixXd measurementNoiseCovSqrt = measurementNoiseCov.cwiseSqrt();

    Eigen::VectorXd state(DP);
    state << 0.1;

    Eigen::VectorXd measurement(MP);

    Mat cv_q( DP, 1, type );
    Mat cv_r( MP, 1, type );
    Eigen::VectorXd q(DP);
    Eigen::VectorXd r(MP);

    Eigen::VectorXd u(DP);

    cv::RNG rng(216);

    state(0) = 10;
    g_x_arr.push_back(state(0));
    g_m_arr.push_back(0);
    g_r_arr.push_back(0);

    for (int i = 0; i < nIterations; i++)
    {
        rng.fill(cv_q, RNG::NORMAL, Scalar::all(0), Scalar::all(1));
        rng.fill(cv_r, RNG::NORMAL, Scalar::all(0), Scalar::all(1));
        cv2eigen(cv_q, q);
        cv2eigen(cv_r, r);
        q = processNoiseCovSqrt * q;
        r = measurementNoiseCovSqrt * r;

        u(0) = (double)i;
        g_u_arr.push_back(u(0));
        g_q_arr.push_back(q(0));

        UnivariateNonstationaryGrowthModel::stateConversionFunction(state(0), u(0), q(0), state(0));

        g_x_arr.push_back(state(0));

        //model->measurementFunction(state, zero, exactMeasurement);
        UnivariateNonstationaryGrowthModel::measurementFunction(state(0), r(0), measurement(0));
        g_m_arr.push_back(measurement(0));
        g_r_arr.push_back(r(0));
    }

    // last data
    g_u_arr.push_back(0);
    g_q_arr.push_back(0);
    

    std::cout << "data prepared " << g_x_arr.size() << std::endl;
}

class TransitionCostFunctor
{
public:
    static ceres::CostFunction* Create() {
        return new ceres::AutoDiffCostFunction<TransitionCostFunctor, 1, 1, 1, 1>(new TransitionCostFunctor());
    }

public:
    TransitionCostFunctor()
    {
    }
    
    template <typename T>
    bool operator()(const T* const param1 , const T* const param2, const T* const param3, T* residuals) const {
    #if 1
        T xi = param1[0];
        T xj = param2[0];
        T u = param3[0];
        T x_next;
        UnivariateNonstationaryGrowthModel::stateConversionFunction(xi, u, T(0), x_next);
        residuals[0] = /*T(1.0) */ (xj - x_next);
        //std::cout << "=== res " << residuals[0] << std::endl;
        return true;
    #else
        residuals[0] = T(0);
        return true;
    #endif
    }
};

class MeasurementCostFunctor
{
public:
    static ceres::CostFunction* Create(const double& m) {
        return new ceres::AutoDiffCostFunction<MeasurementCostFunctor, 1, 1>(new MeasurementCostFunctor(m));
    }

public:
    MeasurementCostFunctor() = delete;
    MeasurementCostFunctor(const double& m):m_(m)
    {
    }
    
    template <typename T>
    bool operator()(const T* const param1, T* residuals) const {
    #if 1
        T x = param1[0];
        T m;
        UnivariateNonstationaryGrowthModel::measurementFunction(x, T(0), m);
        residuals[0] = /*T(1.0) */ (T(m_) - m);
        //std::cout << "=== res " << residuals[0] << std::endl;
        return true;
    #else
        residuals[0] = T(0);
        return true;
    #endif
    }

private:
    double m_;
};

int main(int argc, char **argv)
{
    google::InitGoogleLogging(argv[0]);

    prepare_data(100);

    auto param_x = g_x_arr;
    const int state_size = 1;

    // Build the problem.
    ceres::Problem problem;

    for (int i = 0; i < param_x.size(); ++i) {
        problem.AddParameterBlock(param_x.data()+i, state_size);
        
        problem.AddParameterBlock(g_u_arr.data()+i, state_size);
        problem.AddParameterBlock(g_q_arr.data()+i, state_size);
        problem.AddParameterBlock(g_m_arr.data()+i, state_size);
        problem.AddParameterBlock(g_r_arr.data()+i, state_size);

        problem.SetParameterBlockConstant(g_u_arr.data()+i);
        problem.SetParameterBlockConstant(g_q_arr.data()+i);
        problem.SetParameterBlockConstant(g_m_arr.data()+i);
        problem.SetParameterBlockConstant(g_r_arr.data()+i);
    }

    for (int i = 0; i < param_x.size() - 1; ++i) {
        problem.AddResidualBlock(MeasurementCostFunctor::Create(g_m_arr[i]), NULL, param_x.data()+i);
    }

    for (int i = 0; i < param_x.size() - 1; ++i) {
        int j = i + 1;
        problem.AddResidualBlock(TransitionCostFunctor::Create(), NULL, 
            param_x.data()+i, param_x.data()+j, g_u_arr.data()+i);
    }

    // Run the solver!
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << "\n";

    std::ofstream file_target("testUKF_target.txt");
    std::ofstream file_corrected("testUKF_cor_mean.txt");

    for (int i = 0; i < param_x.size(); ++i) {
        file_target << g_x_arr[i] << std::endl;
        file_corrected << param_x[i] << std::endl;
    }

    return 0;
}

