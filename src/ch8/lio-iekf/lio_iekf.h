#ifndef SAD_CH8_LASER_MAPPING_H
#define SAD_CH8_LASER_MAPPING_H

#include <livox_ros_driver/CustomMsg.h>
#include <pcl/filters/voxel_grid.h>
#include <sensor_msgs/PointCloud2.h>

/// 部分类直接使用ch7的结果
#include "ch3/static_imu_init.h"
#include "ch7/loosely_coupled_lio/cloud_convert.h"
#include "ch7/loosely_coupled_lio/measure_sync.h"
#include "ch7/ndt_inc.h"
#include "ch8/lio-iekf/iekf.hpp"

#include "tools/ui/pangolin_window.h"

#include "ch8/ikd-Tree/icp_inc.h"                             // 用于点面ICP
#include "ch8/ikd-Tree/ikd_Tree.h"                    // 用于构建增量式kd-tree

namespace sad {

class LioIEKF {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    struct Options {
        Options() {}
        bool save_motion_undistortion_pcd_ = false;  // 是否保存去畸变前后的点云
        bool with_ui_ = true;                        // 是否带着UI
        int NUM_MATCH_POINTS = 5;                       // 最近邻点数，默认5个
    };

    LioIEKF(Options options = Options());
    ~LioIEKF() = default;

    /// init without ros
    bool Init(const std::string& config_yaml);

    /// 点云回调函数
    void PCLCallBack(const sensor_msgs::PointCloud2::ConstPtr& msg);
    void LivoxPCLCallBack(const livox_ros_driver::CustomMsg::ConstPtr& msg);

    /// IMU回调函数
    void IMUCallBack(IMUPtr msg_in);

    /// 结束程序，退出UI
    void Finish();

    /// 获取当前姿态
    NavStated GetCurrentState() const { return ieskf_.GetNominalState(); }

    /// 获取当前扫描
    CloudPtr GetCurrentScan() const { return current_scan_; }

   private:
    bool LoadFromYAML(const std::string& yaml_file);

    /// 处理同步之后的IMU和雷达数据
    void ProcessMeasurements(const MeasureGroup& meas);

    /// 尝试让IMU初始化
    void TryInitIMU();

    /// 利用IMU预测状态信息
    /// 这段时间的预测数据会放入imu_states_里
    void Predict();

    /// 对measures_中的点云去畸变
    void Undistort();

    /// 执行一次配准和观测
    void Align();

    /// ICP配准
    void AlignICP_ikdtree();

    /// 局部地图处理和增加
    void MapIncremental();

    /// 类型转换
    CloudPtr PointVecToCloudPtr(const PointVec& pointVec);

    /// modules
    std::shared_ptr<MessageSync> sync_ = nullptr;
    StaticIMUInit imu_init_;

    /// point clouds data
    FullCloudPtr scan_undistort_fullcloud_{new FullPointCloudType()};  // scan after undistortion
    CloudPtr scan_undistort_{new PointCloudType()};
    CloudPtr scan_down_body_{new PointCloudType()};     // 【新增】 雷达系下降采样后的点云
    CloudPtr scan_down_world_{new PointCloudType()};    // 【新增】 世界系下降采样后的点云
    std::vector<PointVec> nearest_points_;              // 【新增】 近邻点
    CloudPtr current_scan_ = nullptr;

    /// NDT数据
    IncNdt3d ndt_;
    SE3 last_pose_;

    // flags
    bool imu_need_init_ = true;
    bool flg_first_scan_ = true;
    double filter_size_map_min_ = 0.5; // 【新增】高博 yaml中的参数为0.5，默认为0
    bool flg_ESKF_inited_ = false;
    int frame_num_ = 0;

    double first_lidar_time_ = 0.0; // 【新增】

    // 【新增】ICP数据
    IncIcp3d icp_;

    ///////////////////////// EKF inputs and output ///////////////////////////////////////////////////////
    MeasureGroup measures_;  // sync IMU and lidar scan
    std::vector<NavStated> imu_states_;
    IESKFD ieskf_;  // IESKF
    SE3 TIL_;       // Lidar与IMU之间外参

    Options options_;
    std::shared_ptr<ui::PangolinWindow> ui_ = nullptr;
};

}  // namespace sad

#endif  // FASTER_LIO_LASER_MAPPING_H