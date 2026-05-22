cmake_minimum_required(VERSION 3.16.0)

project(calib_mono_cam DESCRIPTION "Calibration of monocular camera manually" LANGUAGES CXX C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE "Release")

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-O3 -Wall -std=c++17")

find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)

include_directories(
    sophus
    ${OpenCV_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/inc
    ${PROJECT_SOURCE_DIR}/src
)

find_package(Eigen3 REQUIRED)
message(STATUS ${EIGEN3_VERSION})

add_subdirectory(utility)

add_executable(test_opt_camK test_opt_camK.cpp)
target_link_libraries(test_opt_camK PRIVATE ${CERES_LIBRARIES} ${OpenCV_LIBS})

add_executable(use_quat_optimiztion 
    test_quat.cpp
    src/camera_calib.cpp  
    src/use_quat_parameterization.cpp)
target_link_libraries(use_quat_optimiztion ${OpenCV_LIBS} ${CERES_LIBRARIES})
target_link_libraries(use_quat_optimiztion ultility)
target_compile_definitions(use_quat_optimiztion PRIVATE -DUSE_QUATERNION_PARAMETRIZATION)
set_target_properties(use_quat_optimiztion PROPERTIES OUTPUT_NAME quat_opt RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

add_executable(use_lie_optimiztion 
    test_lie.cpp 
    src/camera_calib.cpp 
    src/use_lie_parameterization.cpp)
target_link_libraries(use_lie_optimiztion ${OpenCV_LIBS} ${CERES_LIBRARIES})
target_link_libraries(use_lie_optimiztion ultility)
target_compile_definitions(use_lie_optimiztion PRIVATE -DUSE_LIE_PARAMETRIZATION)
set_target_properties(use_lie_optimiztion PROPERTIES OUTPUT_NAME lie_opt RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

add_executable(use_numeric_optimiztion
    test_numeric.cpp
    src/camera_calib.cpp 
    src/use_numeric.cpp)
target_link_libraries(use_numeric_optimiztion ${OpenCV_LIBS} ${CERES_LIBRARIES})
target_link_libraries(use_numeric_optimiztion ultility)
target_compile_definitions(use_numeric_optimiztion PRIVATE -DUSE_NUMERIC_DIFF)
set_target_properties(use_numeric_optimiztion PROPERTIES OUTPUT_NAME num_opt RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)



