/*
 * @Author: Jimn gaoqianmain@gmail.com
 * @Date: 2026-05-20 09:17:07
 * @LastEditors: Jimn gaoqianmain@gmail.com
 * @LastEditTime: 2026-05-22 16:33:44
 * @FilePath: /calibration-by-myself/src/camera_calib.cpp
 * @Description: https://blog.csdn.net/weixin_43763292/article/details/128546103?csdn_share_tail=%7B%22type%22%3A%22blog%22%2C%22rType%22%3A%22article%22%2C%22rId%22%3A%22128546103%22%2C%22source%22%3A%22weixin_43763292%22%7D
 */
#include "camera_calib.hpp"
#include <opencv2/core/eigen.hpp>
#include <ceres/rotation.h>
#include <cstring>

using namespace std;
CamCalib::CamCalib(const std::string &pic_path, const int points_per_row, const int points_per_col, const double square_size):
    pic_path_(pic_path),
    points_per_row_(points_per_row),
    points_per_col_(points_per_col),
    square_size_(square_size)
{
    K_ = cv::Mat::eye(3, 3, CV_64F);
    dist_coeff_ = cv::Mat::zeros(4, 1, CV_64F);
}

bool CamCalib::readPictures()
{
    calib_pics_.clear();
    ori_pics_.clear();
    for (int i = 0; i <= 40; ++i)
    {
        std::string single_picture_path = pic_path_ + "/" + std::to_string(i + 100000) + ".png";
        cv::Mat pic = cv::imread(single_picture_path, cv::IMREAD_GRAYSCALE);
        if (pic.empty())
        {
            std::cerr << "read picture failed: " << single_picture_path << std::endl;
            return false;
        }
#ifdef DEBUG
        cv::imshow("origin image", pic);
        cv::waitKey(100);
#endif
        calib_pics_.push_back(pic);
        ori_pics_.push_back(pic);
    }
    std::cout << "ReadPics succeed" << std::endl;
    return true; 
}

bool CamCalib::getKeyPoints(){
    points_3d_vec_.clear();
    points_2d_vec_.clear();
    
    // 世界坐标系下的点
    for (size_t i = 0; i < calib_pics_.size(); ++i){
        std::vector<cv::Point2f> points_3ds;
        for (int row = 0; row < points_per_row_; ++row){
            for (int col = 0; col < points_per_col_; ++col){
                cv::Point2f corner_w(row * square_size_, col * square_size_);
                points_3ds.push_back(corner_w);
            }
        }
        points_3d_vec_.push_back(std::move(points_3ds));
    }

    for (const auto &calib_pic: calib_pics_)
    {
        std::vector<cv::Point2f> corner_pts;
        bool found_flag = cv::findChessboardCorners(
            calib_pic, cv::Size(points_per_col_, points_per_row_), corner_pts,
            cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE); //!!! cv::Size(col,row)
        if (!found_flag)
        {
            std::cerr << "found chessborad corner failed";
            return false;
        }
        cv::TermCriteria criteria = cv::TermCriteria(
            cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001);
        cv::cornerSubPix(calib_pic, corner_pts, cv::Size(11, 11), cv::Size(-1, -1),
                         criteria);
        #ifdef DEBUG
            // 角点绘制
            cv::drawChessboardCorners(calib_pic, cv::Size(points_per_row_, points_per_col_), corner_pts, found_flag);
            cv::imshow("chessboard corner image", calib_pic);
            cv::waitKey(300);
        #endif
        points_2d_vec_.push_back(std::move(corner_pts));
    }

    std::cout << "GetKeyPoints succeed" << std::endl;
    return true;
}

bool CamCalib::calcH()
{
    for(size_t id = 0; id < points_2d_vec_.size(); ++id)
    {
        // 获取单幅图像的像素和3d坐标点
        const auto &points_2d = points_2d_vec_[id];
        const auto &points_3d = points_3d_vec_[id];

        cv::Mat H = cv::Mat::eye(3, 3, CV_64F);
        int corner_size = points_3d.size();
        if (corner_size < 4)
        { 
            std::cerr << "corner size less than 4" << std::endl;
            exit(-1);
        }

        cv::Mat A(corner_size * 2, 9, CV_64F, cv::Scalar(0));
        for (int i = 0; i < corner_size; ++i)
        {
            const auto& p_3d = points_3d[i];
            const auto& p_2d = points_2d[i];
        
            A.at<double>(2 * i, 0) = p_3d.x;
            A.at<double>(2 * i, 1) = p_3d.y;
            A.at<double>(2 * i, 2) = 1;
            A.at<double>(2 * i, 3) = 0;
            A.at<double>(2 * i, 4) = 0;
            A.at<double>(2 * i, 5) = 0;
            A.at<double>(2 * i, 6) = -p_2d.x * p_3d.x;
            A.at<double>(2 * i, 7) = -p_2d.x * p_3d.y;
            A.at<double>(2 * i, 8) = -p_2d.x;

            A.at<double>(2 * i + 1, 0) = 0;
            A.at<double>(2 * i + 1, 1) = 0;
            A.at<double>(2 * i + 1, 2) = 0;
            A.at<double>(2 * i + 1, 3) = p_3d.x;
            A.at<double>(2 * i + 1, 4) = p_3d.y;
            A.at<double>(2 * i + 1, 5) = 1;
            A.at<double>(2 * i + 1, 6) = -p_2d.y * p_3d.x;
            A.at<double>(2 * i + 1, 7) = -p_2d.y * p_3d.y;
            A.at<double>(2 * i + 1, 8) = -p_2d.y;
        }
        cv::Mat U, W, Vt; //A = UWV^T
        cv::SVD::compute(A, W, U, Vt, cv::SVD::FULL_UV | cv::SVD::MODIFY_A);
        H = Vt.row(8).reshape(0, 3);

        if (W.at<double>(8) > 0)
        {
            H_vec_.push_back(H);
        }
    }
    return true;
}

void CamCalib::calcK()
{
    cv::Mat A(points_2d_vec_.size() * 2, 6, CV_64F, cv::Scalar(0)); // 计算约束矩阵B
    
    std::cout << "H size is: " << H_vec_.size() << std::endl;
    for (size_t id = 0; id < points_2d_vec_.size(); ++id)
    {
        cv::Mat H = H_vec_[id];
        // 第1列
        double h11 = H.at<double>(0, 0);
        double h21 = H.at<double>(1, 0);
        double h31 = H.at<double>(2, 0);
        // 第2列
        double h12 = H.at<double>(0, 1);
        double h22 = H.at<double>(1, 1);
        double h32 = H.at<double>(2, 1);
        
        cv::Mat v11 = (cv::Mat_<double>(1, 6) << 
            h11 * h11, h11 * h21 + h21 * h11, 
            h21 * h21, h11 * h31 + h31 * h11,
            h21 * h31 + h31 * h21, h31 * h31
        );

        cv::Mat v12 = (cv::Mat_<double>(1, 6) << 
            h11 * h12, h11 * h22 + h21 * h12, 
            h21 * h22, h11 * h32 + h31 * h12, 
            h21 * h32 + h31 * h22, h31 * h32
        );

        cv::Mat v22 = (cv::Mat_<double>(1, 6) << 
            h12 * h12, h12 * h22 + h22 * h12, 
            h22 * h22, h12 * h32 + h32 * h12,
            h22 * h32 + h32 * h22, h32 * h32
        );

        v12.copyTo(A.row(2 * id));
        cv::Mat v_tmp = v11 - v22;
        v_tmp.copyTo(A.row(2 * id + 1));
    }
    cv::Mat U, W, Vt;
    cv::SVD::compute(A, W, U, Vt);
    cv::Mat B = Vt.row(5); // 得到B
    std::cout << "B: " << std::endl << B << std::endl;

    double B11 = B.at<double>(0, 0);
    double B12 = B.at<double>(0, 1);
    double B22 = B.at<double>(0, 2);
    double B13 = B.at<double>(0, 3);
    double B23 = B.at<double>(0, 4);
    double B33 = B.at<double>(0, 5);

    double v0 = (B12 * B13 - B11 * B23) / (B11 * B22 - B12 * B12);
    double lambda = B33 - (B13 * B13 + v0 * (B12 * B13 - B11 * B23)) / B11;
    double alpha = sqrt(lambda / B11);
    double beta = sqrt(lambda * B11 / (B11 * B22 - B12 * B12));
    double gamma = -B12 * alpha * alpha * beta / lambda;
    double u0 = gamma * v0 / beta - B13 * alpha * alpha / lambda;
    std::cout << "K coeff: " << alpha << " , " << beta << " , " << gamma << " , " << lambda << " , " << u0 << " , " << v0 << std::endl;
    gamma = 0;
    K_ = (cv::Mat_<double>(3, 3) << alpha, gamma, u0, 0, beta, v0, 0, 0, 1);
    std::cout << "K: " << std::endl << K_ << std::endl;
}

void CamCalib::calcRT()
{
    cv::Mat K_inv;
    cv::invert(K_, K_inv);
    for (const auto& H: H_vec_){
        cv::Mat R_t = K_inv * H;
        // 利用旋转矩阵的列向量模长为1，求出缩放系数
        double scale = 1 / ((cv::norm(R_t.col(0)) + cv::norm(R_t.col(1))) / 2);
        if(R_t.at<double>(2, 2) < 0) { // 需要确保深度是正的，作用是保证坐标系都是统一，Z轴和镜头同向
            R_t = -R_t;
        }
        R_t = R_t * scale;
        cv::Vec3d r1(R_t.at<double>(0, 0), R_t.at<double>(1, 0), R_t.at<double>(2, 0));
        cv::Vec3d r2(R_t.at<double>(0, 1), R_t.at<double>(1, 1), R_t.at<double>(2, 1));
        cv::Vec3d r3 = r1.cross(r2);
        cv::Mat Q = cv::Mat::eye(3, 3, CV_64F);
        Q.at<double>(0, 0) = r1(0);
        Q.at<double>(1, 0) = r1(1);
        Q.at<double>(2, 0) = r1(2);
        Q.at<double>(0, 1) = r2(0);
        Q.at<double>(1, 1) = r2(1);
        Q.at<double>(2, 1) = r2(2);
        Q.at<double>(0, 2) = r3(0);
        Q.at<double>(1, 2) = r3(1);
        Q.at<double>(2, 2) = r3(2);
        cv::Mat U, W, VT;                                                         // A =UWV^T
        cv::SVD::compute(Q, W, U, VT, cv::SVD::MODIFY_A | cv::SVD::FULL_UV); // Eigen 返回的是V,列向量就是特征向量, opencv 返回的是VT，所以行向量是特征向量
        cv::Mat R = U * VT;
        double det = cv::determinant(R);
        if (det < 0) {
            R.col(2) *= -1;
        }
        R_vec_.push_back(R);
        cv::Mat R_T;
        cv::transpose(R, R_T);
        cv::Mat t = cv::Mat::eye(3, 1, CV_64F);
        R_t.col(2).copyTo(t.col(0));
        t_vec_.push_back(t);
        std::cout << "t: " << t.t() << std::endl;
    }
}

void CamCalib::calcDistCoeff()
{
    std::vector<double> r2_vec;
    std::vector<cv::Point2f> ideal_point_vec;
    
    for (size_t id = 0; id < points_3d_vec_.size(); id++){
        ideal_point_vec.clear();
        r2_vec.clear();
        cv::Mat R = R_vec_[id];
        cv::Mat t = t_vec_[id];
        std::vector<cv::Point2f> point_3ds = points_3d_vec_[id];
        for (const auto &p : point_3ds)
        {
            cv::Mat p_3d = (cv::Mat_<double>(3, 1) << p.x, p.y, 0);
            cv::Mat p_pic = R * p_3d + t;
            p_pic.at<double>(0, 0) = p_pic.at<double>(0, 0) / p_pic.at<double>(2, 0);
            p_pic.at<double>(1, 0) = p_pic.at<double>(1, 0) / p_pic.at<double>(2, 0);
            p_pic.at<double>(2, 0) = 1;
            double x = p_pic.at<double>(0, 0);
            double y = p_pic.at<double>(1, 0);
            double r2 = x * x + y * y;
            r2_vec.push_back(r2);

            cv::Mat p_uv = K_ * p_pic;
            ideal_point_vec.emplace_back(p_uv.at<double>(0, 0), p_uv.at<double>(1, 0));
        }

        // points_2d_vec_ is distort uv
        std::vector<cv::Point2f> dist_point_vec = points_2d_vec_[id];
        double u0 = K_.at<double>(0, 2);
        double v0 = K_.at<double>(1, 2);
        cv::Mat D = cv::Mat::zeros(ideal_point_vec.size() * 2, 2, CV_64F);
        cv::Mat d = cv::Mat::zeros(ideal_point_vec.size() * 2, 1, CV_64F);
        for (size_t i = 0; i < ideal_point_vec.size(); ++i)
        {
            double r2 = r2_vec[i];
            cv::Point2f distort_p = dist_point_vec[i];
            cv::Point2f ideal_p = ideal_point_vec[i];
            D.at<double>(2 * i, 0) = (ideal_p.x - u0) * r2;
            D.at<double>(2 * i, 1) = (ideal_p.x - u0) * r2 * r2;
            D.at<double>(2 * i + 1, 0) = (ideal_p.y - v0) * r2;
            D.at<double>(2 * i + 1, 1) = (ideal_p.y - v0) * r2 * r2;
            d.at<double>(0, 0) = distort_p.x - ideal_p.x;
            d.at<double>(1, 0) = distort_p.y - ideal_p.y;
        }

        cv::Mat DT;
        cv::transpose(D, DT);
        cv::Mat DTD_inverse;
        cv::invert(DT * D, DTD_inverse);
        dist_coeff_ += DTD_inverse * DT * d;
        
        
    }
    dist_coeff_ /= points_3d_vec_.size();
    std::cout << "distort coeff: " << dist_coeff_.at<double>(0, 0) << ", " << dist_coeff_.at<double>(1, 0) << std::endl;
}

void CamCalib::visualizeCameraPoses(){
    if (R_vec_.empty() || t_vec_.empty()){
        printf("\033[31m[ERROR] R_vec_ or t_vec_ is empty!\033[0m\n]]");
        return;
    }

    int canvas_width = 1280;
    int canvas_height = 720;
    cv::Mat canvas_top = cv::Mat::zeros(canvas_height, canvas_width, CV_8UC3);
    cv::Mat canvas_side = cv::Mat::zeros(canvas_height, canvas_width, CV_8UC3);

    canvas_top.setTo(cv::Scalar(255, 255, 255));
    canvas_side.setTo(cv::Scalar(255, 255, 255));

    double max_dist = 0.0;
    cv::Point2d center_offset(canvas_width / 2.0, canvas_height / 2.0);

    for(const auto& t: t_vec_){
        double dist = cv::norm(t);
        max_dist = std::max(max_dist, dist);
    }

    double scale = std::min(canvas_width, canvas_height) / (2.0 * max_dist);

    cv::line(canvas_top, cv::Point(0, canvas_height / 2), cv::Point(canvas_width, canvas_height / 2), cv::Scalar(200, 200, 200), 1);
    cv::line(canvas_top, cv::Point(canvas_width / 2, 0), cv::Point(canvas_width/2, canvas_height), cv::Scalar(200, 200, 200), 1);
    cv::line(canvas_side, cv::Point(0, canvas_height / 2), cv::Point(canvas_width, canvas_height / 2), cv::Scalar(200, 200, 200), 1);
    cv::line(canvas_side, cv::Point(canvas_width / 2, 0), cv::Point(canvas_width /2 , canvas_height), cv::Scalar(200, 200, 200), 1);

    // 绘制每个相机的位置和方向
    for (size_t i = 0; i < R_vec_.size(); i++){
        const cv::Mat& R = R_vec_[i];
        const cv::Mat& t = t_vec_[i];

        // 相机在世界坐标系中的位置
        cv::Mat R_t;
        cv::transpose(R, R_t);
        cv::Mat camera_pos = -R_t * t;

        double cam_x = camera_pos.at<double>(0, 0);
        double cam_y = camera_pos.at<double>(1, 0);
        double cam_z = camera_pos.at<double>(2, 0);

        cv::Point2d pt_top(center_offset.x + cam_x * scale, center_offset.y - cam_y * scale);
        cv::Point2d pt_side(center_offset.x + cam_x * scale, center_offset.y - cam_z * scale);

        // 绘制相机位置
        cv::circle(canvas_top, pt_top, 5, cv::Scalar(255, 0, 0), -1);
        cv::circle(canvas_side, pt_side, 5, cv::Scalar(255, 0, 0), -1);

        double axis_length = 0.2 * scale;
        cv::Mat x_axis = (cv::Mat_<double>(3, 1) << 1, 0, 0);
        cv::Mat y_axis = (cv::Mat_<double>(3, 1) << 0, 1, 0);
        cv::Mat z_axis = (cv::Mat_<double>(3, 1) << 0, 0, 1);

        // 世界坐标下的相机的三轴的
        cv::Mat x_world = R_t * x_axis;
        cv::Mat y_world = R_t * y_axis;
        cv::Mat z_world = R_t * z_axis;

        cv::line(canvas_top, pt_top, cv::Point(pt_top.x + axis_length * x_world.at<double>(0), pt_top.y - axis_length * x_world.at<double>(1)), cv::Scalar(0, 0, 255), 2); // x轴
        cv::line(canvas_top, pt_top, cv::Point(pt_top.x + axis_length * y_world.at<double>(0), pt_top.y - axis_length * y_world.at<double>(1)), cv::Scalar(0, 255, 0), 2); // y轴
        cv::line(canvas_top, pt_top, cv::Point(pt_top.x + axis_length * z_world.at<double>(0), pt_top.y - axis_length * z_world.at<double>(1)), cv::Scalar(255, 0, 0), 2); // z轴

        cv::line(canvas_side, pt_side, cv::Point(pt_side.x + axis_length * x_world.at<double>(0), pt_side.y - axis_length * x_world.at<double>(1)), cv::Scalar(0, 0, 255), 2);
        cv::line(canvas_side, pt_side, cv::Point(pt_side.x + axis_length * y_world.at<double>(0), pt_side.y - axis_length * y_world.at<double>(1)), cv::Scalar(0, 255, 0), 2);
        cv::line(canvas_side, pt_side, cv::Point(pt_side.x + axis_length * z_world.at<double>(0), pt_side.y - axis_length * z_world.at<double>(1)), cv::Scalar(255, 0, 0), 2);

        // 相机注册编号
        std::string label = std::to_string(i);
        cv::putText(canvas_top, label, cv::Point(pt_top.x + 10, pt_top.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
        cv::putText(canvas_side, label, cv::Point(pt_side.x + 10, pt_side.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
    }

    // 绘制原点，标定板位置
    cv::circle(canvas_top, center_offset, 8, cv::Scalar(0, 0, 0), -1);
    cv::circle(canvas_side, center_offset, 8, cv::Scalar(0, 0, 0), -1);
    cv::putText(canvas_top, "origin", cv::Point(center_offset.x + 10, center_offset.y + 20), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0,0,0));
    cv::putText(canvas_side, "origin", cv::Point(center_offset.x + 10, center_offset.y + 20), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0,0,0));

    cv::imshow("top", canvas_top);
    cv::imshow("side", canvas_side);

    cv::waitKey(0);
}

void CamCalib::saveCameraPoses(const std::string& filename){
    std::ofstream fout(filename);
    if (!fout.is_open()){
        std::cout << "save camera poses failed" << std::endl;
        return;
    }

    fout << "# Camera Poses (R and t)" << std::endl;
    fout << "Format: id R00, R01, R02, R10, R11, R12, R20, R21, R22, tx, ty, tz" << std::endl;
    fout << "No. of poses: " << R_vec_.size() << std::endl;
    fout << std::endl;

    for (size_t i = 0; i < R_vec_.size(); i++)
    {
        const cv::Mat& R = R_vec_[i];
        const cv::Mat& t = t_vec_[i];

        fout << i << " ";
        for (int r = 0; r < 3; r++)
        {
            for (int c = 0; c < 3; c++)
            {
                fout << R.at<double>(r, c) << " ";
            }
        }

        fout << t.at<double>(0) << " " << t.at<double>(1) << " " << t.at<double>(2) << std::endl;
    }

    fout.close();
    std::cout << "Camera Pose saved to: " << filename << std::endl;
}