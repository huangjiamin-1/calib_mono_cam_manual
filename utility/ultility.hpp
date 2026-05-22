/*
 * @Author: Jimn gaoqianmain@gmail.com
 * @Date: 2026-05-22 09:49:47
 * @LastEditors: Jimn gaoqianmain@gmail.com
 * @LastEditTime: 2026-05-22 09:58:18
 * @FilePath: /calib_mono_cam_manual/src/ultility.hpp
 * @Description: 工具函数
 */
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

Eigen::Matrix3d skew(const Eigen::Vector3d &v);

template <typename Derived>
Eigen::Quaternion<typename Derived::Scalar> deltaQ(const Eigen::MatrixBase<Derived> &theta)
{
    typedef typename Derived::Scalar Scalar_t;

    Eigen::Quaternion<Scalar_t> dq;
    Eigen::Matrix<Scalar_t, 3, 1> half_theta = theta;
    half_theta /= static_cast<Scalar_t>(2.0);
    dq.w() = static_cast<Scalar_t>(1.0);
    dq.x() = half_theta.x();
    dq.y() = half_theta.y();
    dq.z() = half_theta.z();
    return dq;
}
