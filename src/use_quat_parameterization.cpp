#include "../inc/camera_calib.hpp"
#include "pose_local_parameteriztion.hpp"
#include "so3.hpp"

/// @brief 这个是使用Eigen实现的四元数 注意使用ceres自带的四元数比较多坑
class MonoCamFactor : public ceres::SizedCostFunction<2, 4, 3, 4, 4>
{ 
    public:
        MonoCamFactor(const Eigen::Vector2d &uv, const Eigen::Vector3d &point_3d):
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
            Eigen::Quaterniond q(parameters[0][3], parameters[0][0], parameters[0][1], parameters[0][2]);
            Eigen::Matrix3d R = q.toRotationMatrix();
            // ceres::QuaternionRotatePoint()
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
                    Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor>> jaco_uv2Dtheta(jacobians[0]);
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

void CamCalib::optimiztion_quat()
{
    printf("\033[32m============================================== \n");
    printf("使用四元数进行优化\n");
    printf("注意Eigen使用顺序是[x,y,z,w]但是赋值的时候是[w,x,y,z]=>自己继承实现LocalParameterization\n");
    printf("但是ceres中使用的是[w,x,y,z], 所以最好使用Eigen的方式\n");
    printf("在ceres的Evaluate中可以使用ceres::QuaternionRotatePoint()来进行Rp操作\n");
    printf("============================================== \033[0m\n");
    ceres::Problem problem;
    ceres::LossFunction * loss;
    loss = new ceres::HuberLoss(1.0);
    int pic_num = points_3d_vec_.size();
    double cam_param_R[pic_num][4];
    double cam_param_t[pic_num][3];

    for (int i = 0; i < pic_num; ++i){
        const cv::Mat &R = R_vec_[i];
        const cv::Mat &t = t_vec_[i];
        cv::Mat angle_axis; // 计算角轴 李代数
        cv::Rodrigues(R, angle_axis);

        Eigen::Matrix3d Reigen = Sophus::SO3d::exp(Eigen::Vector3d(angle_axis.at<double>(0, 0), angle_axis.at<double>(1, 0), angle_axis.at<double>(2, 0))).matrix();
        Eigen::Quaterniond q(Reigen);
        cam_param_R[i][0] = q.x(); 
        cam_param_R[i][1] = q.y();
        cam_param_R[i][2] = q.z();
        cam_param_R[i][3] = q.w();

        cam_param_t[i][0] = t.at<double>(0, 0);
        cam_param_t[i][1] = t.at<double>(1, 0);
        cam_param_t[i][2] = t.at<double>(2, 0);

        ceres::LocalParameterization* local = new PoseLocalParameterization();
        problem.AddParameterBlock(cam_param_R[i], 4, local);
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
            auto* factor = new MonoCamFactor(Eigen::Vector2d(pt2d.x, pt2d.y), Eigen::Vector3d(pt3d.x, pt3d.y, 0));
            
            problem.AddResidualBlock(factor, nullptr, cam_param_R[i], cam_param_t[i], K_param, coeff_param);
        }
    }

    // return;
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    // options.max_num_iterations = 1000;
    // options.max_solver_time_in_seconds = 10;
    // options.parameter_tolerance = 1e-9; 
    // options.function_tolerance = 1e-8; 
    options.minimizer_progress_to_stdout = true;
    
    // options.trust_region_strategy_type = ceres::DOGLEG;
    
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;
    // std::cout << summary.BriefReport() << std::endl;

    std::cout << "origin K:\n"
              << K_ << std::endl;
    std::cout << "origin dist_coeff\n:" << dist_coeff_ << std::endl;
    K_.at<double>(0, 0) = K_param[0];
    K_.at<double>(1, 1) = K_param[1];
    K_.at<double>(0, 2) = K_param[2];
    K_.at<double>(1, 2) = K_param[3];
    dist_coeff_ = cv::Mat(4, 1, CV_64F);
    dist_coeff_.at<double>(0, 0) = coeff_param[0];
    dist_coeff_.at<double>(1, 0) = coeff_param[1];
    dist_coeff_.at<double>(2, 0) = coeff_param[2];
    dist_coeff_.at<double>(3, 0) = coeff_param[3];

    printf("\033[32m====================================================================\n");
    printf("out:\n");
    printf("K: [fx, fy, cx, cy]=[%f, %f, %f, %f]\n", K_param[0], K_param[1], K_param[2], K_param[3]);
    printf("distort: [k1, k2, p1, p2]=[%f, %f, %f, %f]\n", coeff_param[0], coeff_param[1], coeff_param[2], coeff_param[3]);
    printf("====================================================================\033[0m\n");
    
    delete[] coeff_param;
}
