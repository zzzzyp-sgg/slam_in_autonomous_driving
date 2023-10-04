#ifndef SLAM_IN_AUTO_DRIVING_ICP_INC_3D_H
#define SLAM_IN_AUTO_DRIVING_ICP_INC_3D_H

// #include "ch5/kdtree.h"
#include "common/eigen_types.h"
#include "common/point_types.h"
#include "ch8/ikd-Tree/ikd_Tree.h"    // 【新增】用于构建增量式kd-tree，来自fast-lio

#include <glog/logging.h>

namespace sad {

class IncIcp3d {
   public:
    struct Options {
        int max_iteration_ = 20;                // 最大迭代次数
        double max_plane_distance_ = 0.05;      // 平面最近邻查找时阈值
        int min_effective_pts_ = 10;            // 最近邻点数阈值
        double eps_ = 1e-2;                     // 收敛判定条件
        bool use_initial_translation_ = false;  // 是否使用初始位姿中的平移估计
    };

    IncIcp3d() {}
    IncIcp3d(Options options) : options_(options) {}

    /// 设置目标的Scan
    void SetTarget(CloudPtr target) {
        target_ = target;
        BuildTargetKdTree();

        // 计算点云中心
        target_center_ = std::accumulate(target->points.begin(), target_->points.end(), Vec3d::Zero().eval(),
                                         [](const Vec3d& c, const PointType& pt) -> Vec3d { return c + ToVec3d(pt); }) /
                         target_->size();
        LOG(INFO) << "target center: " << target_center_.transpose();
    }

    /// 设置被配准的Scan
    void SetSource(CloudPtr source) {
        source_ = source;
        source_center_ = std::accumulate(source_->points.begin(), source_->points.end(), Vec3d::Zero().eval(),
                                         [](const Vec3d& c, const PointType& pt) -> Vec3d { return c + ToVec3d(pt); }) /
                         source_->size();
        LOG(INFO) << "source center: " << source_center_.transpose();
    }

    void ComputeResidualAndJacobians_P2Plane(const SE3& input_pose, Mat18d& HTVH, Vec18d& HTVr);

private:
    // 建立目标点云的Kdtree
    void BuildTargetKdTree();

    // std::shared_ptr<KdTree> kdtree_ = nullptr;  // 第5章的kd树
    std::shared_ptr<KD_TREE<PointType>> ikdtreePtr = nullptr;  // fast-lio中的增量式kd树

    CloudPtr target_ = nullptr;
    CloudPtr source_ = nullptr;

    Vec3d target_center_ = Vec3d::Zero();
    Vec3d source_center_ = Vec3d::Zero();

    Options options_;
};

}  // namespace sad

#endif  // SLAM_IN_AUTO_DRIVING_ICP_INC_3D_H