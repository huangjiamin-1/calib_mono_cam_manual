/*
 * @Author: Jimn gaoqianmain@gmail.com
 * @Date: 2026-05-20 09:09:29
 * @LastEditors: Jimn gaoqianmain@gmail.com
 * @LastEditTime: 2026-05-22 14:41:53
 * @FilePath: /calibration-by-myself/main.cpp
 * @Description: 
 */

#include "camera_calib.hpp"
int main(int argc, char const *argv[])
{
    std::string pic_path = "../data";
    CamCalib cam_intr_calib(pic_path, 11, 8, 0.02);    
    cam_intr_calib.readPictures();
    cam_intr_calib.getKeyPoints();
    cam_intr_calib.calcH();
    cam_intr_calib.calcK();
    cam_intr_calib.calcRT();
    cam_intr_calib.calcDistCoeff();
    cam_intr_calib.optimiztion_lie();
    cam_intr_calib.visualizeCameraPoses();
    cam_intr_calib.saveCameraPoses("../data/camera_poses_lie.txt");
    return 0;
}
