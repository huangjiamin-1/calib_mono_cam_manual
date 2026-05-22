/*
 * @Author: Jimn gaoqianmain@gmail.com
 * @Date: 2026-05-22 09:50:54
 * @LastEditors: Jimn gaoqianmain@gmail.com
 * @LastEditTime: 2026-05-22 09:50:55
 * @FilePath: /calib_mono_cam_manual/src/ultility.cpp
 * @Description: 工具函数实现
 */
#include "ultility.hpp"

Eigen::Matrix3d skew(const Eigen::Vector3d &v)
{ 
    Eigen::Matrix3d skew_mat;
    skew_mat << 0, -v(2), v(1),
                v(2), 0, -v(0),
                -v(1), v(0), 0;
    return skew_mat;
}