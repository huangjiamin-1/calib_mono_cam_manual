
#include "camera_calib.hpp"
#include "so3_local_param.hpp"
#include "ultility.hpp"
class MonoCamFactorLie : public ceres::SizedCostFunction<2, 3, 3, 4, 4>
{ 
    public:
        MonoCamFactorLie(const Eigen::Vector2d &uv, const Eigen::Vector3d &point_3d):
        uv_(uv), point_3d_(point_3d)
        {}

        virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
        {
            // 将内参表示为矩阵
            double fx = parameters[2][0];
            double fy = parameters[2][1];
            double cx = parameters[2][2];
            double cy = parameters[2][3]; 

            Eigen::Vector3d t(parameters[1][0], parameters[1][1], parameters[1][2]);
            Eigen::Matrix3d R = Sophus::SO3d::exp(Eigen::Vector3d(parameters[0][0], parameters[0][1], parameters[0][2])).matrix();
            // ceres::AngleAxisRotatePoint() 也是ok的
            Eigen::Vector3d Rp = R * point_3d_;
            double k1 = parameters[3][0];
            double k2 = parameters[3][1];
            double p1 = parameters[3][2];
            double p2 = parameters[3][3];

            auto pt_3d = Rp + t;
            double x = pt_3d.x();
            double y = pt_3d.y();
            double z = pt_3d.z();
            double z2 = z * z;
            if (fabs(z) < 1e-8) {
                residuals[0] = residuals[1] = 1e6; // 残差设为大值，跳过无效点
                return true;
            }
            
            auto norm_plane = pt_3d / z; // 归一化平面

            double u_normal = norm_plane.x();
            double v_normal = norm_plane.y();

            double u_normal2 = u_normal * u_normal;
            double v_normal2 = v_normal * v_normal;

            double u_normal4 = u_normal2 * u_normal2;
            double v_normal4 = v_normal2 * v_normal2;
            double u_normal3 = u_normal2 * u_normal;
            double v_normal3 = v_normal2 * v_normal;
            double uv_normal = u_normal * v_normal;

            // cv::calibrateCamera()

            double r2 = u_normal2 + v_normal2; // r^2
            double r4 = r2 * r2;

            double u_distorted = u_normal * (1 + k1 * r2 + k2 * r4) + 2 * p1 * uv_normal + p2 * (r2 + 2 * u_normal2);
            double v_distorted = v_normal * (1 + k1 * r2 + k2 * r4) + 2 * p2 * uv_normal + p1 * (r2 + 2 * v_normal2);

            double u_est = u_distorted * fx + cx;
            double v_est = v_distorted * fy + cy;
            residuals[0] = uv_[0] - u_est;
            residuals[1] = uv_[1] - v_est;
            // printf("\033[32m %f, %f \033[0m\n", residuals[0], residuals[1]);

            if (jacobians)
            {
                Eigen::Matrix2d J_uv2uv_distort_normal = Eigen::Matrix2d::Identity(); // 像素对畸变的归一化坐标的Jacobian
                J_uv2uv_distort_normal(0, 0) = -fx;
                J_uv2uv_distort_normal(1, 1) = -fy;

                Eigen::Matrix<double, 2, 3> jaco_uv_normal2xyz = Eigen::Matrix<double, 2, 3>::Zero();
                jaco_uv_normal2xyz(0, 0) = 1 / z;
                jaco_uv_normal2xyz(0, 2) = -x / z2;
                jaco_uv_normal2xyz(1, 1) = 1 / z;
                jaco_uv_normal2xyz(1, 2) = -y / z2;

                Eigen::Matrix2d jaco_uv_distort_noraml2uv_normal;
                jaco_uv_distort_noraml2uv_normal(0, 0) = 1 + 3 * k1 * u_normal2 + k1 * v_normal2 + 5 * k2 * u_normal4 + k2 * v_normal4 + 6 * k2 * u_normal2 * v_normal2 + 2 * p1 * v_normal + 6 * p2 * u_normal;
                jaco_uv_distort_noraml2uv_normal(0, 1) = 2 * k1 * uv_normal + 4 * k2 * u_normal * v_normal3 + 4 * k2 * u_normal3 * v_normal + 2 * p1 * u_normal + 2 * p2 * v_normal;
                jaco_uv_distort_noraml2uv_normal(1, 0) = 2 * k1 * uv_normal + 4 * k2 * v_normal * u_normal3 + 4 * k2 * u_normal * v_normal3 + 2 * p1 * u_normal + 2 * p2 * v_normal;
                jaco_uv_distort_noraml2uv_normal(1, 1) = 1 + k1 * u_normal2 + 3 * k1 * v_normal2 + k2 * u_normal4 + 5 * k2 * v_normal4 + 6 * k2 * u_normal2 * v_normal2 + 6 * p1 * v_normal + 2 * p2 * u_normal;
                
                if(jacobians[0]){
                    Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> jaco_uv2Dtheta(jacobians[0]);
                    jaco_uv2Dtheta.setZero();
                    // uv对扰动theta的Jacobian
                    jaco_uv2Dtheta.block<2, 3>(0, 0) = -J_uv2uv_distort_normal * jaco_uv_distort_noraml2uv_normal * jaco_uv_normal2xyz * skew(Rp);
                }

                if(jacobians[1]){
                    Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> jaco_uv2T(jacobians[1]);
                    jaco_uv2T.setZero();
                    // uv对位移的Jacobian
                    jaco_uv2T.block<2, 3>(0, 0) = J_uv2uv_distort_normal * jaco_uv_distort_noraml2uv_normal * jaco_uv_normal2xyz;
                }

                if(jacobians[2]){
                    Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor>> jaco_uv2fxfycxcy(jacobians[2]);
                    jaco_uv2fxfycxcy.setZero();
                    jaco_uv2fxfycxcy(0, 0) = -u_distorted;
                    jaco_uv2fxfycxcy(0, 2) = -1;
                    jaco_uv2fxfycxcy(1, 1) = -v_distorted;
                    jaco_uv2fxfycxcy(1, 3) = -1;
                } 

                if(jacobians[3]){
                    Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor>> jaco_uv2k1k2(jacobians[3]);
                    Eigen::Matrix<double, 2, 4> jaco_k1k2;
                    jaco_k1k2.setZero();
                    jaco_k1k2(0, 0) = r2 * u_normal;
                    jaco_k1k2(0, 1) = r4 * u_normal;
                    jaco_k1k2(0, 2) = 2 * uv_normal;
                    jaco_k1k2(0, 3) = r2 + 2 * u_normal2;
                    jaco_k1k2(1, 0) = r2 * v_normal;
                    jaco_k1k2(1, 1) = r4 * v_normal;
                    jaco_k1k2(1, 2) = r2 + 2 * v_normal2;
                    jaco_k1k2(1, 3) = 2 * uv_normal;
                    jaco_uv2k1k2 = J_uv2uv_distort_normal * jaco_k1k2;
                }
            }
            
            return true;
        }

    private:
        Eigen::Vector2d uv_;
        Eigen::Vector3d point_3d_;
};

void CamCalib::optimiztion_lie()
{
    printf("\033[32m============================================== \n");
    printf("使用angle_axis或李代数进行优化\n");
    printf("可以在ceres的Evaluate中使用ceres::AngleAxisRotatePoint()来进行Rp操作\n");
    printf("============================================== \033[0m\n");

    ceres::Problem problem;
    ceres::LossFunction * loss;
    loss = new ceres::HuberLoss(0.2);
    int pic_num = points_3d_vec_.size();
    double cam_param_R[pic_num][3];
    double cam_param_t[pic_num][3];

    for (int i = 0; i < pic_num; ++i){
        const cv::Mat &R = R_vec_[i];
        const cv::Mat &t = t_vec_[i];
        cv::Mat angle_axis; // 计算角轴 李代数
        cv::Rodrigues(R, angle_axis);

        cam_param_R[i][0] = angle_axis.at<double>(0, 0);
        cam_param_R[i][1] = angle_axis.at<double>(1, 0);
        cam_param_R[i][2] = angle_axis.at<double>(2, 0);
        cam_param_t[i][0] = t.at<double>(0, 0);
        cam_param_t[i][1] = t.at<double>(1, 0);
        cam_param_t[i][2] = t.at<double>(2, 0);

        ceres::LocalParameterization* local = new SO3Parameterization();
        problem.AddParameterBlock(cam_param_R[i], 3, local);
        problem.AddParameterBlock(cam_param_t[i], 3);
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
            auto* factor = new MonoCamFactorLie(Eigen::Vector2d(pt2d.x, pt2d.y), Eigen::Vector3d(pt3d.x, pt3d.y, 0));
            
            problem.AddResidualBlock(factor, loss, cam_param_R[i], cam_param_t[i], K_param, coeff_param);
        }
    }


    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    
    // options.trust_region_strategy_type = ceres::DOGLEG;
    
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;
    // std::cout << summary.BriefReport() << std::endl;

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
    std::cout << points_3d_vec_[0][10] << std::endl;

    double average_error = 0;
    for (int i = 0; i < pic_num; i++)
    {
        double cost = 0;
        cv::Vec3d rvec(cam_param_R[i][0], cam_param_R[i][1], cam_param_R[i][2]);
        cv::Rodrigues(rvec, R_vec_[i]);
        t_vec_[i].at<double>(0, 0) = cam_param_t[i][0];
        t_vec_[i].at<double>(1, 0) = cam_param_t[i][1];
        t_vec_[i].at<double>(2, 0) = cam_param_t[i][2];

        std::cout << "t_vec_[i]" << t_vec_[i].t() << std::endl;
        
        for (int k = 0; k < (int)points_3d_vec_[i].size(); k++)
        {
            cv::Mat point3d = cv::Mat::zeros(3, 1, CV_64FC1);
            point3d.at<double>(0, 0) = points_3d_vec_[i][k].x;
            point3d.at<double>(1, 0) = points_3d_vec_[i][k].y;
            // std::cout << "point3d: " << points_3d_vec_[i][j].x << std::endl;
            auto p = R_vec_[i] * point3d + t_vec_[i];
            double z = cv::Mat(p).at<double>(2, 0);
            double x = cv::Mat(p).at<double>(0, 0) / z;
            double y = cv::Mat(p).at<double>(1, 0) / z;
            double k1 = dist_coeff_.at<double>(0, 0);
            double k2 = dist_coeff_.at<double>(1, 0);
            double p1 = dist_coeff_.at<double>(2, 0);
            double p2 = dist_coeff_.at<double>(3, 0);

            double r2 = x * x + y * y;
            double u = x * (1 + k1 * r2 + k2 * r2 * r2) + p1 * 2 * x * y + p2 * (r2 + 2 * x * x);
            double v = y * (1 + k1 * r2 + k2 * r2 * r2) + p1 * (r2 + 2 * y * y) + p2 * 2 * x * y;

            cv::Mat pp = (cv::Mat_<double>(3, 1) << u, v, 1);
            auto uv = cv::Mat(K_ * pp);
            auto uv_real = points_2d_vec_[i][k];
            x = uv.at<double>(0, 0) - uv_real.x;
            y = uv.at<double>(1, 0) - uv_real.y;
            cost += sqrt(x * x + y * y);
        }
        average_error += cost;

        cost /= points_2d_vec_[i].size();
        // printf("cost: %f\n", cost);
    }
    printf("average cost: %f\n", average_error / points_2d_vec_.size() / points_2d_vec_[0].size());
    delete[] coeff_param;
}