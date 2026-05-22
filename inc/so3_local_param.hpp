/*
 * @Author: Jimn gaoqianmain@gmail.com
 * @Date: 2026-05-22 10:10:29
 * @LastEditors: Jimn gaoqianmain@gmail.com
 * @LastEditTime: 2026-05-22 10:21:27
 * @FilePath: /calib_mono_cam_manual/inc/so3_local_param.hpp
 * @Description: 这是一个关于李代数进行更新局部参数的类
 */

#pragma once
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <eigen3/Eigen/Core>
#include <iostream>
#include <vector>

#include "so3.hpp"

class SO3Parameterization : public ceres::LocalParameterization {
public:
    virtual ~SO3Parameterization() {}
    virtual bool Plus(const double* x, const double* delta, double* x_plus_delta) const {
        Eigen::Map<const Eigen::Vector3d> lie(x);
        Eigen::Map<const Eigen::Vector3d> d_lie(delta);
        Eigen::Map<Eigen::Vector3d> lie_plus(x_plus_delta);
        
        Sophus::SO3d SO3_x      = Sophus::SO3d::exp(lie);
        Sophus::SO3d SO3_delta  = Sophus::SO3d::exp(d_lie);
        // 使用左乘运算更新相机系
        Sophus::SO3d SO3_x_plus = SO3_delta * SO3_x;
        // 执行公式(11)得到真正更新的量
        // 扰动模型                  加法运算 更新公式         也就是输出的结果
        lie_plus = SO3_x_plus.log();    // 因为我们在Evaluate中已经计算过lie了，那是对扰动的导数，更新的话需要R^3空间中计算，所以需要将 Exp(d_lie) *
                                        // Exp(lie) => Exp(Jl * d_lie + lie) => Jl * d_lie + lie
        return true;
    }

    virtual bool ComputeJacobian(const double* x, double* jacobian) const {
        // Eigen::Map<const Eigen::Vector3d> lie(x);
        Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J(jacobian);
        J.setIdentity();    // 3x3单位矩阵
        // J = Sophus::SO3::exp(lie).inverse().Adj();    // 这里不能使用左雅可比矩阵的逆进行计算 
        return true;
    }

    virtual int GlobalSize() const {
        return 3;
    }
    virtual int LocalSize() const {
        return 3;
    }
};
