#include <pcl/common/transforms.h>
#include <yaml-cpp/yaml.h>
#include <execution>
#include <fstream>

#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/sparse_block_matrix.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>

// #include "ch4/g2o_types_preinteg.h"
#include "common/g2o_types.h"

#include "common/lidar_utils.h"
#include "common/point_cloud_utils.h"
#include "common/timer/timer.h"

#include "lio_iekf.h"

namespace sad {

LioIEKF::LioIEKF(Options options) : options_(options) {
    StaticIMUInit::Options imu_init_options;
    imu_init_options.use_speed_for_static_checking_ = false;  // 本节数据不需要轮速计
    imu_init_ = StaticIMUInit(imu_init_options);
}

bool LioIEKF::Init(const std::string &config_yaml) {
    if (!LoadFromYAML(config_yaml)) {
        LOG(INFO) << "init failed.";
        return false;
    }

    if (options_.with_ui_) {
        ui_ = std::make_shared<ui::PangolinWindow>();
        ui_->Init();
    }

    return true;
}

void LioIEKF::ProcessMeasurements(const MeasureGroup &meas) {
    LOG(INFO) << "call meas, imu: " << meas.imu_.size() << ", lidar pts: " << meas.lidar_->size();
    measures_ = meas;

    if (imu_need_init_) {
        // 初始化IMU系统
        TryInitIMU();
        return;
    }

    // 利用IMU数据进行状态预测
    Predict();

    // 对点云去畸变
    Undistort();

    // 配准
    // Align();
    AlignICP_ikdtree();
}

bool LioIEKF::LoadFromYAML(const std::string &yaml_file) {
    // get params from yaml
    sync_ = std::make_shared<MessageSync>([this](const MeasureGroup &m) { ProcessMeasurements(m); });
    sync_->Init(yaml_file);

    /// 自身参数主要是雷达与IMU外参
    auto yaml = YAML::LoadFile(yaml_file);
    std::vector<double> ext_t = yaml["mapping"]["extrinsic_T"].as<std::vector<double>>();
    std::vector<double> ext_r = yaml["mapping"]["extrinsic_R"].as<std::vector<double>>();

    Vec3d lidar_T_wrt_IMU = math::VecFromArray(ext_t);
    Mat3d lidar_R_wrt_IMU = math::MatFromArray(ext_r);
    TIL_ = SE3(lidar_R_wrt_IMU, lidar_T_wrt_IMU);
    return true;
}

void LioIEKF::Align() {
    FullCloudPtr scan_undistort_trans(new FullPointCloudType);
    pcl::transformPointCloud(*scan_undistort_fullcloud_, *scan_undistort_trans, TIL_.matrix().cast<float>());
    scan_undistort_fullcloud_ = scan_undistort_trans;

    current_scan_ = ConvertToCloud<FullPointType>(scan_undistort_fullcloud_);

    // voxel
    pcl::VoxelGrid<PointType> voxel;
    voxel.setLeafSize(0.5, 0.5, 0.5);
    voxel.setInputCloud(current_scan_);

    CloudPtr current_scan_filter(new PointCloudType);
    voxel.filter(*current_scan_filter);

    /// the first scan
    if (flg_first_scan_) {
        ndt_.AddCloud(current_scan_);
        flg_first_scan_ = false;

        return;
    }

    // 后续的scan，使用NDT配合pose进行更新
    LOG(INFO) << "=== frame " << frame_num_;

    ndt_.SetSource(current_scan_filter);
    // 这里进行了IMU以及NDT的更新操作
    ieskf_.UpdateUsingCustomObserve([this](const SE3 &input_pose, Mat18d &HTVH, Vec18d &HTVr) {
        ndt_.ComputeResidualAndJacobians(input_pose, HTVH, HTVr);
    });

    auto current_nav_state = ieskf_.GetNominalState();

    // 若运动了一定范围，则把点云放入地图中
    SE3 current_pose = ieskf_.GetNominalSE3();
    SE3 delta_pose = last_pose_.inverse() * current_pose;

    if (delta_pose.translation().norm() > 1.0 || delta_pose.so3().log().norm() > math::deg2rad(10.0)) {
        // 将地图合入NDT中
        CloudPtr current_scan_world(new PointCloudType);
        pcl::transformPointCloud(*current_scan_filter, *current_scan_world, current_pose.matrix());
        ndt_.AddCloud(current_scan_world);
        last_pose_ = current_pose;
    }

    // 放入UI
    if (ui_) {
        ui_->UpdateScan(current_scan_, current_nav_state.GetSE3());  // 转成Lidar Pose传给UI
        ui_->UpdateNavState(current_nav_state);
    }

    frame_num_++;
    return;
}

void LioIEKF::TryInitIMU() {
    for (auto imu : measures_.imu_) {
        imu_init_.AddIMU(*imu);
    }

    if (imu_init_.InitSuccess()) {
        // 读取初始零偏，设置ESKF
        sad::IESKFD::Options options;
        // 噪声由初始化器估计
        options.gyro_var_ = sqrt(imu_init_.GetCovGyro()[0]);
        options.acce_var_ = sqrt(imu_init_.GetCovAcce()[0]);
        ieskf_.SetInitialConditions(options, imu_init_.GetInitBg(), imu_init_.GetInitBa(), imu_init_.GetGravity());
        imu_need_init_ = false;

        LOG(INFO) << "IMU初始化成功";
    }
}

void LioIEKF::Undistort() {
    auto cloud = measures_.lidar_;
    auto imu_state = ieskf_.GetNominalState();  // 最后时刻的状态
    SE3 T_end = SE3(imu_state.R_, imu_state.p_);

    if (options_.save_motion_undistortion_pcd_) {
        sad::SaveCloudToFile("./data/ch7/before_undist.pcd", *cloud);
    }

    /// 将所有点转到最后时刻状态上
    std::for_each(std::execution::par_unseq, cloud->points.begin(), cloud->points.end(), [&](auto &pt) {
        SE3 Ti = T_end;
        NavStated match;

        // 根据pt.time查找时间，pt.time是该点打到的时间与雷达开始时间之差，单位为毫秒
        math::PoseInterp<NavStated>(
            measures_.lidar_begin_time_ + pt.time * 1e-3, imu_states_, [](const NavStated &s) { return s.timestamp_; },
            [](const NavStated &s) { return s.GetSE3(); }, Ti, match);

        Vec3d pi = ToVec3d(pt);
        Vec3d p_compensate = TIL_.inverse() * T_end.inverse() * Ti * TIL_ * pi;

        pt.x = p_compensate(0);
        pt.y = p_compensate(1);
        pt.z = p_compensate(2);
    });
    scan_undistort_fullcloud_ = cloud;

    if (options_.save_motion_undistortion_pcd_) {
        sad::SaveCloudToFile("./data/ch7/after_undist.pcd", *cloud);
    }
}

void LioIEKF::Predict() {
    imu_states_.clear();
    imu_states_.emplace_back(ieskf_.GetNominalState());

    /// 对IMU状态进行预测
    for (auto &imu : measures_.imu_) {
        ieskf_.Predict(*imu);
        imu_states_.emplace_back(ieskf_.GetNominalState());
    }
}

void LioIEKF::PCLCallBack(const sensor_msgs::PointCloud2::ConstPtr &msg) { sync_->ProcessCloud(msg); }

void LioIEKF::LivoxPCLCallBack(const livox_ros_driver::CustomMsg::ConstPtr &msg) { sync_->ProcessCloud(msg); }

void LioIEKF::IMUCallBack(IMUPtr msg_in) { sync_->ProcessIMU(msg_in); }

void LioIEKF::Finish() {
    if (ui_) {
        ui_->Quit();
    }
    LOG(INFO) << "finish done";
}

float calc_dist(pcl::Vector3fMapConst a, Eigen::Vector3f b)
{
    float dist = 0.0f;
    dist = (a.x() - b.x()) * (a.x() - b.x()) + (a.y() - b.y()) * (a.y() - b.y()) + (a.z() - b.z()) * (a.z() - b.z());
    return dist;
}

float calc_dist(pcl::Vector3fMap a, Eigen::Vector3f b)
{
    float dist = 0.0f;
    dist = (a.x() - b.x()) * (a.x() - b.x()) + (a.y() - b.y()) * (a.y() - b.y()) + (a.z() - b.z()) * (a.z() - b.z());
    return dist;
}

/**
 * @description: 仿照增量式NDT，新建一个icp_inc_3d.cc文件，实现增量式ICP
 * @return {*}
 */
void LioIEKF::AlignICP_ikdtree() {
    FullCloudPtr scan_undistort_trans(new FullPointCloudType);
    pcl::transformPointCloud(*scan_undistort_fullcloud_, *scan_undistort_trans, TIL_.matrix().cast<float>());
    scan_undistort_fullcloud_ = scan_undistort_trans;

    scan_undistort_ = ConvertToCloud<FullPointType>(scan_undistort_fullcloud_);

    // 点云降采样
    pcl::VoxelGrid<PointType> voxel;
    voxel.setLeafSize(0.5, 0.5, 0.5);
    voxel.setInputCloud(scan_undistort_);
    voxel.filter(*scan_down_body_); // 体素滤波，降采样

    /// the first scan
    if (flg_first_scan_) {
        // ndt_.AddCloud(scan_undistort_);
        icp_.SetTarget(scan_undistort_);  // 【新增】

        first_lidar_time_ = measures_.lidar_begin_time_;
        flg_first_scan_ = false;
        return;
    }

    // 后续的scan，使用NDT配合pose进行更新
    LOG(INFO) << "=== frame " << frame_num_;

    int cur_pts = scan_down_body_->size(); // 降采样后的去畸变点云数量

    // ndt_.SetSource(scan_down_body_);
    icp_.SetSource(scan_down_body_); // 【新增】为点面icp中的ikdtree设置原始点云
    ieskf_.UpdateUsingCustomObserve([this](const SE3 &input_pose, Mat18d &HTVH, Vec18d &HTVr) {
                                        // ndt_.ComputeResidualAndJacobians(input_pose, HTVH, HTVr);
                                        icp_.ComputeResidualAndJacobians_P2Plane(input_pose, HTVH, HTVr); // 【新增】计算点面残差和雅可比
                                    });

    auto current_nav_state = ieskf_.GetNominalState();

    // 若运动了一定范围，则把点云放入地图中
    SE3 current_pose = ieskf_.GetNominalSE3();
    SE3 delta_pose = last_pose_.inverse() * current_pose;

    if (delta_pose.translation().norm() > 1.0 || delta_pose.so3().log().norm() > math::deg2rad(10.0)) {
        // 将地图合入NDT中
        CloudPtr scan_down_world_2(new PointCloudType);
        pcl::transformPointCloud(*scan_down_body_, *scan_down_world_2, current_pose.matrix());
        // ndt_.AddCloud(scan_down_world_2);
        icp_.SetTarget(scan_down_world_2); // 【新增】为点面icp中的ikdtree设置目标点云，内部际是添加新的点云到ikdtree中
        last_pose_ = current_pose;
    }

    // 放入UI
    if (ui_) {
        ui_->UpdateScan(scan_undistort_, current_nav_state.GetSE3());  // 转成Lidar Pose传给UI
        ui_->UpdateNavState(current_nav_state);
    }

    frame_num_++;
    return;
}

CloudPtr LioIEKF::PointVecToCloudPtr(const PointVec& pointVec) {
    CloudPtr cloud(new PointCloudType);

    // 遍历 PointVec 并将点添加到 CloudPtr
    for (const PointType& point : pointVec) {
        cloud->points.push_back(point);
    }

    // 设置点云的宽度和高度（如果需要）
    cloud->width = cloud->points.size();
    cloud->height = 1;  // 1 表示非结构化点云

    return cloud;
}

void LioIEKF::MapIncremental() {
    PointVec points_to_add;
    PointVec point_no_need_downsample;

    int cur_pts = scan_down_body_->size();
    points_to_add.reserve(cur_pts);
    point_no_need_downsample.reserve(cur_pts);

    std::vector<size_t> index(cur_pts);
    for (size_t i = 0; i < cur_pts; ++i) 
        index[i] = i;

    // 并发处理
    std::for_each(  std::execution::unseq, 
                    index.begin(), index.end(), 
                    [&](const size_t &i) {
                        /* transform to world frame */
                        // 雷达系转换到世界系
                        // PointBodyToWorld(&(scan_down_body_->points[i]), &(scan_down_world_->points[i]));

                        /* decide if need add to map */
                        // 判断是否需要加入到局部地图中
                        PointType &point_world = scan_down_world_->points[i];

                        // 判断第i个点的近邻点集是否为空
                        if (!nearest_points_[i].empty() && flg_ESKF_inited_) {
                            // 取出第i个点的近邻点集
                            const PointVec &points_near = nearest_points_[i];

                            // 计算中心坐标
                            Eigen::Vector3f center = ((point_world.getVector3fMap() / filter_size_map_min_).array().floor() + 0.5) * filter_size_map_min_;

                            // 计算第i个点到中心点的L1距离
                            Eigen::Vector3f dis_2_center = points_near[0].getVector3fMap() - center;

                            // 判断距离是否大于阈值
                            if (fabs(dis_2_center.x()) > 0.5 * filter_size_map_min_ &&
                                fabs(dis_2_center.y()) > 0.5 * filter_size_map_min_ &&
                                fabs(dis_2_center.z()) > 0.5 * filter_size_map_min_) {
                                // 若是，则加入到无需降采样点集中
                                point_no_need_downsample.emplace_back(point_world);
                                return; // 程序返回？因为这里是lambda函数内部，所以返回的是lambda函数，而不是MapIncremental函数
                            }

                            // 此时，标记改为需要增加
                            bool need_add = true;
                            // 计算第i个点到中心点的L2距离
                            float dist = calc_dist(point_world.getVector3fMap(), center); // 【在math_utils.h】中添加了两个函数实现
                            // 判断近邻点数是否多于5个
                            if (points_near.size() >= options_.NUM_MATCH_POINTS) {
                                // 遍历所有近邻点
                                for (int readd_i = 0; readd_i < options_.NUM_MATCH_POINTS; readd_i++) {
                                    // 判断这些近邻点距离中心点的距离是否小于阈值
                                    if (calc_dist(points_near[readd_i].getVector3fMap(), center) < dist + 1e-6) {
                                        need_add = false; // 只要有一个距离很小的，就不需要增加了，直接跳出循环
                                        break;
                                    }
                                }
                            }
                            // 判断是否需要增加
                            if (need_add) 
                                // 加入到需要增加的点集中
                                points_to_add.emplace_back(point_world);
                        } else 
                            points_to_add.emplace_back(point_world);
                    });

    LOG(INFO) << "points_to_add.size: " << points_to_add.size() << " point_no_need_downsample.size: " << point_no_need_downsample.size();
    icp_.SetTarget(PointVecToCloudPtr(points_to_add));            // 【新增】为点面icp中的ikdtree设置目标点云，内部际是添加新的点云到ikdtree中
    icp_.SetTarget(PointVecToCloudPtr(point_no_need_downsample)); // 【新增】为点面icp中的ikdtree设置目标点云，内部际是添加新的点云到ikdtree中
}

}  // namespace sad