
#include "opencv2/tracking/kalman_filters.hpp"
#include <iostream>
#include <fstream>

using namespace cv;

class UnivariateNonstationaryGrowthModel: public cv::tracking::UkfSystemModel
{

public:
    void stateConversionFunction(const Mat& x_k, const Mat& u_k, const Mat& v_k, Mat& x_kplus1)
    {
        double x = x_k.at<double>(0, 0);
        double n = u_k.at<double>(0, 0);
        double q = v_k.at<double>(0, 0);
        double u = u_k.at<double>(0, 0);

        double x1 = 0.5*x + 25*( x/(x*x + 1) ) + 8*cos( 1.2*(n-1) ) + q;
        x_kplus1.at<double>(0, 0) = x1;
    }
    void measurementFunction(const Mat& x_k, const Mat& n_k, Mat& z_k)
    {
        double x = x_k.at<double>(0, 0);
        double r = n_k.at<double>(0, 0);

        double y = x + r;//x*x/20.0 + r;
        z_k.at<double>(0, 0) = y;
    }
};


int main()
{
    const double alpha = 1.0;
    const double beta = 2.0;
    const double kappa = 0.0;

    const double mse_treshold = 0.5;
    const int nIterations = 100; // number of observed iterations

    int MP = 1;
    int DP = 1;
    int CP = 0;
    int type = CV_64F;

    Ptr<UnivariateNonstationaryGrowthModel> model( new UnivariateNonstationaryGrowthModel() );
    cv::tracking::UnscentedKalmanFilterParams params( DP, MP, CP, 0, 0, model );

    Mat processNoiseCov = Mat::zeros( DP, DP, type );
    processNoiseCov.at<double>(0, 0) = 1.0;
    Mat processNoiseCovSqrt = Mat::zeros( DP, DP, type );
    sqrt( processNoiseCov, processNoiseCovSqrt );

    Mat measurementNoiseCov = Mat::zeros( MP, MP, type );
    measurementNoiseCov.at<double>(0, 0) = 1.0;
    Mat measurementNoiseCovSqrt = Mat::zeros( MP, MP, type );
    sqrt( measurementNoiseCov, measurementNoiseCovSqrt );

    Mat P = Mat::eye( DP, DP, type );

    Mat state( DP, 1, type );
    state.at<double>(0, 0) = 0.1;

    Mat initState = state.clone();
    initState.at<double>(0, 0) = 0.0;

    params.errorCovInit = P;
    params.measurementNoiseCov = measurementNoiseCov;
    params.processNoiseCov = processNoiseCov;
    params.stateInit = initState.clone();

    params.alpha = alpha;
    params.beta = beta;
    params.k = kappa;

    Mat correctStateAUKF( DP, 1, type );

    Mat measurement( MP, 1, type );
    Mat exactMeasurement( MP, 1, type );

    Mat q( DP, 1, type );
    Mat r( MP, 1, type );

    Mat u( DP, 1, type );
    Mat zero = Mat::zeros( MP, 1, type );

    RNG rng( 216 );

    double average_error = 0.0;
    const int repeat_count = 1;
    for (int j = 0; j<repeat_count; j++)
    {
        std::ofstream file_target("testUKF_target.txt");
        std::ofstream file_corrected("testUKF_cor_mean.txt");
        cv::Ptr<cv::tracking::UnscentedKalmanFilter> uncsentedKalmanFilter = cv::tracking::createUnscentedKalmanFilter( params );
        state.at<double>(0, 0) = 10;

        double mse = 0.0;
        for (int i = 0; i<nIterations; i++)
        {
            rng.fill( q, RNG::NORMAL, Scalar::all(0), Scalar::all(1) );
            rng.fill( r, RNG::NORMAL, Scalar::all(0), Scalar::all(1) );
            q = processNoiseCovSqrt*q;
            r = measurementNoiseCovSqrt*r;

            u.at<double>(0, 0) = (double)i;
            model->stateConversionFunction(state, u, q, state);

            //model->measurementFunction(state, zero, exactMeasurement);
            model->measurementFunction(state, r, measurement);

            uncsentedKalmanFilter->predict( u );
            correctStateAUKF = uncsentedKalmanFilter->correct( measurement );

            mse +=  pow( state.at<double>(0, 0) - correctStateAUKF.at<double>(0, 0), 2.0 );
            file_target << state.at<double>(0, 0) << std::endl;
            file_corrected << correctStateAUKF.at<double>(0, 0) << std::endl;
        }
        mse /= nIterations;
        average_error += mse;
    }
    average_error /= repeat_count;

    //assert( mse_treshold > average_error );
    std::cout << "finish with average_error " << average_error << std::endl;
    return 0;
}
