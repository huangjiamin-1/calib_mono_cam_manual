#include "camera_calib.hpp"

struct ReprojErr
{
public:
    ReprojErr(const cv::Point2f& uv, const cv::Point2f& world_3d):
    observe_p_2d(uv), world_p_3d(world_3d)
    {}

    template<typename T>
    bool operator()(const T *const camera, const T *const K, const T *const dist_coeff, T *residual) const{
        T p_3d[3] = {static_cast<T>(world_p_3d.x), static_cast<T>(world_p_3d.y), static_cast<T>(0)};
        T p[3];
        ceres::AngleAxisRotatePoint(camera, p_3d, p);
        // camera[3,4,5] are the translation.
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];
        // Compute the center of distortion. The sign change comes from
        // the camera model that Noah Snavely's Bundler assumes, whereby
        // the camera coordinate system has a negative z axis.
        T x = p[0] / p[2];
        T y = p[1] / p[2];
        T r2 = x * x + y * y;

        const T &alpha = K[0];
        const T &beta = K[1];
        const T &u0 = K[2];
        const T &v0 = K[3];

        const T &k1 = dist_coeff[0];
        const T &k2 = dist_coeff[1];
        const T &p1 = dist_coeff[2];
        const T &p2 = dist_coeff[3];

        T x_dist = x * (static_cast<T>(1) + k1 * r2 + k2 * r2 * r2) + T(2) * p1 * x * y + p2 * (r2 + T(2) * x * x);
        T y_dist = y * (static_cast<T>(1) + k1 * r2 + k2 * r2 * r2) + T(2) * p2 * x * y + p1 * (r2 + T(2) * y * y);

        const T u_dist = alpha * x_dist + u0;
        const T v_dist = beta * y_dist + v0;

        residual[0] = u_dist - static_cast<T>(observe_p_2d.x);
        residual[1] = v_dist - static_cast<T>(observe_p_2d.y);
        return true;
    }

    static ceres::CostFunction *Create(const cv::Point2f &observe_p_2d_, const cv::Point2f &world_p_3d_)
    {
        return new ceres::AutoDiffCostFunction<ReprojErr, 2, 6, 4, 4>(new ReprojErr(observe_p_2d_, world_p_3d_));
    }

    const cv::Point2f observe_p_2d;
    const cv::Point2f world_p_3d;
};

void CamCalib::optimiztion_numeric()
{
    ceres::Problem problem;
    ceres::LossFunction * loss;
    loss = new ceres::HuberLoss(1.0);
    int pic_num = points_3d_vec_.size();
    double cam_param[pic_num][6];

    for (int i = 0; i < pic_num; ++i){
        const cv::Mat &R = R_vec_[i];
        const cv::Mat &t = t_vec_[i];
        cv::Mat angle_axis;
        cv::Rodrigues(R, angle_axis);

        cam_param[i][0] = angle_axis.at<double>(0, 0);
        cam_param[i][1] = angle_axis.at<double>(1, 0);
        cam_param[i][2] = angle_axis.at<double>(2, 0);
        cam_param[i][3] = t.at<double>(0, 0);
        cam_param[i][4] = t.at<double>(1, 0);
        cam_param[i][5] = t.at<double>(2, 0);
    }

    for (int i = 0; i < pic_num; ++i){
        problem.AddParameterBlock(cam_param[i], 6);
    }

    double K_param[4];
    K_param[0] = K_.at<double>(0, 0);
    K_param[1] = K_.at<double>(1, 1);
    K_param[2] = K_.at<double>(0, 2);
    K_param[3] = K_.at<double>(1, 2);
    problem.AddParameterBlock(K_param, 4);
    double *coeff_param = new double[4];
    coeff_param[0] = dist_coeff_.at<double>(0, 0);
    coeff_param[1] = dist_coeff_.at<double>(1, 0);
    coeff_param[2] = 0.0;
    coeff_param[3] = 0.0;
    problem.AddParameterBlock(coeff_param, 4);

    for (int i = 0; i < pic_num; ++i)
    {
        const std::vector<cv::Point2f> &points_3ds = points_3d_vec_[i];
        const std::vector<cv::Point2f> &points_2ds = points_2d_vec_[i];
        for (size_t j = 0; j < points_3ds.size(); ++j)
        {
            const auto& pt3d = points_3ds[j];
            const auto& pt2d = points_2ds[j];
            auto* factor = ReprojErr::Create(pt2d, pt3d);
            problem.AddResidualBlock(factor, nullptr, cam_param[i], K_param, coeff_param);
        }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    
    // options.trust_region_strategy_type = ceres::DOGLEG;
    
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;

    printf("\033[32m======================================================================================================================================\n");
    printf("这是numeric优化结果:\n");
    printf("out:\n");
    printf("original K: [fx, fy, cx, cy]=[%f, %f, %f, %f]\n", K_.at<double>(0, 0), K_.at<double>(1, 1), K_.at<double>(0, 2), K_.at<double>(1, 2));
    printf("original dist: [k1, k2, p1, p2]=[%f, %f, %f, %f]\n", dist_coeff_.at<double>(0, 0), dist_coeff_.at<double>(1, 0), 0.0, 0.0);
    printf("K: [fx, fy, cx, cy]=[%f, %f, %f, %f]\n", K_param[0], K_param[1], K_param[2], K_param[3]);
    printf("distort: [k1, k2, p1, p2]=[%f, %f, %f, %f]\n", coeff_param[0], coeff_param[1], coeff_param[2], coeff_param[3]);
    printf("======================================================================================================================================\033[0m\n");

    K_.at<double>(0, 0) = K_param[0];
    K_.at<double>(1, 1) = K_param[1];
    K_.at<double>(0, 2) = K_param[2];
    K_.at<double>(1, 2) = K_param[3];
    dist_coeff_ = cv::Mat(4, 1, CV_64F);
    dist_coeff_.at<double>(0, 0) = coeff_param[0];
    dist_coeff_.at<double>(1, 0) = coeff_param[1];
    dist_coeff_.at<double>(2, 0) = coeff_param[2];
    dist_coeff_.at<double>(3, 0) = coeff_param[3];
    
    delete[] coeff_param;
}