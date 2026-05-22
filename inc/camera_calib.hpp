/*
 * @Author: Jimn gaoqianmain@gmail.com
 * @Date: 2026-05-20 09:17:07
 * @LastEditors: Jimn gaoqianmain@gmail.com
 * @LastEditTime: 2026-05-22 14:37:13
 * @FilePath: /calibration-by-myself/src/camera_calib.hpp
 * @Description: 张正友标定参考: https://zhuanlan.zhihu.com/p/94244568
 */

#pragma once
#include <ceres/ceres.h>
#include "ceres/rotation.h"
#include <iostream>
#include <string>
#include <vector>
#include <opencv4/opencv2/opencv.hpp>

class CamCalib { 
    public:
        explicit CamCalib(const std::string &pic_path, const int points_per_row,
                const int points_per_col,
                const double square_size);
        
        bool readPictures();

        bool getKeyPoints();
        
        /// @brief 计算本质矩阵
        /// @return true=success, false=fail
        bool calcH();

        /// @brief 计算相机内参
        void calcK();

        /// @brief 计算相机外参
        void calcRT();

        /// @brief 计算畸变系数
        void calcDistCoeff();

#ifdef USE_LIE_PARAMETRIZATION
        /// @brief 使用李代数
        void optimiztion_lie();
#endif 

#ifdef USE_QUATERNION_PARAMETRIZATION
        /// @brief 使用四元数
        void optimiztion_quat();
#endif 

#ifdef USE_NUMERIC_DIFF
        /// @brief 使用数值微分
        void optimiztion_numeric();
#endif

        void visualizeCameraPoses();

        void saveCameraPoses(const std::string& filename);
        
    private:
        const std::string pic_path_;
        const int points_per_row_;
        const int points_per_col_;
        const double square_size_;

        std::vector<cv::Mat> calib_pics_;
        std::vector<cv::Mat> ori_pics_;

        cv::Mat K_;
        cv::Mat dist_coeff_;

        std::vector<cv::Mat> H_vec_;
        std::vector<cv::Mat> R_vec_;
        std::vector<cv::Mat> t_vec_;

        std::vector<std::vector<cv::Point2f>> points_3d_vec_; // z=0
        std::vector<std::vector<cv::Point2f>> points_2d_vec_;
};



