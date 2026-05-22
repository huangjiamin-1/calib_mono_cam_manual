/*
 * @Author: Jimn gaoqianmain@gmail.com
 * @Date: 2026-05-20 19:38:51
 * @LastEditors: Jimn gaoqianmain@gmail.com
 * @LastEditTime: 2026-05-22 10:01:57
 * @FilePath: /calibration-by-myself/src/local_ceres.hpp
 * @Description: 这是一个用于更新四元数的 ceres 局部参数化类
 */
#pragma once
#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include "../utility/ultility.hpp"

class PoseLocalParameterization : public ceres::LocalParameterization
{
    virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const
    {
        Eigen::Map<const Eigen::Quaterniond> _q(x);

        Eigen::Quaterniond dq = deltaQ(Eigen::Map<const Eigen::Vector3d>(delta));

        Eigen::Map<Eigen::Quaterniond> q(x_plus_delta);
        q = (dq * _q).normalized();

        return true;
    }
    virtual bool ComputeJacobian(const double *x, double *jacobian) const
    {
        Eigen::Map<Eigen::Matrix<double, 4, 3, Eigen::RowMajor>> j(jacobian);
        j.setZero();
        j.topRows<3>().setIdentity();
        return true;
    }
    virtual int GlobalSize() const { return 4; };
    virtual int LocalSize() const { return 3; };
};