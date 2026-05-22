/*
 * @Author: Jimn gaoqianmain@gmail.com
 * @Date: 2026-05-22 10:26:04
 * @LastEditors: Jimn gaoqianmain@gmail.com
 * @LastEditTime: 2026-05-22 10:33:02
 * @FilePath: /calib_mono_cam_manual/test_opt_camK.cpp
 * @Description: 这是一个验证内参优化的测试文件
 */
#include <ceres/ceres.h>
#include <ceres/cost_function.h>
#include <eigen3/Eigen/Core>

class Factor: public ceres::SizedCostFunction<2, 4>
{
public:
    Factor(const Eigen::Vector3d& uv, const Eigen::Vector3d& pt):uv_(uv), pt_(pt){

    }

    virtual bool Evaluate(double const* const* parameters, double *residuals, double **jacobians) const{
        double fx = parameters[0][0];
        double fy = parameters[0][1];
        double cx = parameters[0][2];
        double cy = parameters[0][3];

        residuals[0] = fx * pt_.x() + cx - uv_.x();
        residuals[1] = fy * pt_.y() + cy - uv_.y();
        

        if (jacobians != nullptr)
        {
            if (jacobians[0] != nullptr)
            {
                Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor>> jaco(jacobians[0]);
                jaco << pt_.x(), 0, 1, 0,
                    0, pt_.y(), 0, 1;
            }
        }
        return true;
    }

    Eigen::Vector3d uv_;
    Eigen::Vector3d pt_;
};

int main(int argc, char const *argv[])
{
    printf("\033[32m===================================================================\n");
    printf("测试优化内参，雅可比矩阵计算\n");
    printf("===================================================================\033[0m\n");
    std::vector<Eigen::Vector3d> data_vec;
    std::vector<Eigen::Vector3d> data_vec_true;
    Eigen::Matrix3d K;
    Eigen::Matrix3d K_false;
    K << 900, 0, 450, 0, 900, 450, 0, 0, 1;
    K_false << 900 + 100, 0, 450 - 8, 0, 900 + 10, 450 - 18, 0, 0, 1;
    for (int i = 0; i < 1000; i++)
    {
        Eigen::Vector3d pt(i * 0.1, i * 0.2, 1);
        auto out_false = K_false * pt;
        auto out = K * pt;
        data_vec.emplace_back(out_false);
        data_vec_true.emplace_back(out);
    }

    double K_[4] = {K_false(0,0), K_false(1,1), K_false(0,2), K_false(1,2)};

    ceres::Problem problem;
    problem.AddParameterBlock(K_, 4);
    for (size_t i = 0; i < 1000; i++)
    {
        Eigen::Vector3d pt(i * 0.1, i * 0.2, 1);
        auto* factor = new Factor(data_vec_true[i], pt);
        problem.AddResidualBlock(factor, nullptr, K_);
    }

    ceres::Solver::Options options;
    // options.linear_solver_type = ceres::DENSE_SCHUR;
    // options.max_num_iterations = 1000;
    // options.max_solver_time_in_seconds = 10;
    options.minimizer_progress_to_stdout = true;    
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << std::endl;
    printf("\033[32m===================================================================\n");
    printf("origin K: [%f, %f, %f, %f]\n", K_false(0, 0), K_false(1, 1), K_false(0, 2), K_false(1, 2));
    printf("ceres K: [%f, %f, %f, %f]\n", K_[0], K_[1], K_[2], K_[3]);
    printf("===================================================================\033[0m\n");

    return 0;
}
