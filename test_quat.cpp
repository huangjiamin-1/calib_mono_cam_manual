/*
 * @Author: Jimn gaoqianmain@gmail.com
 * @Date: 2026-05-20 09:09:29
 * @LastEditors: Jimn gaoqianmain@gmail.com
 * @LastEditTime: 2026-05-22 10:33:57
 * @FilePath: /calibration-by-myself/main.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
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
    cam_intr_calib.optimiztion_quat();
    return 0;
}
