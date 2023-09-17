//
// Created by xiang on 2022/3/22.
//

#ifndef SLAM_IN_AUTO_DRIVING_G2O_TYPES_H
#define SLAM_IN_AUTO_DRIVING_G2O_TYPES_H

#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>

#include <glog/logging.h>
#include <opencv2/core.hpp>

#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/search/impl/kdtree.hpp>
#include "ch6/icp_2d.h"

#include "common/eigen_types.h"
#include "common/math_utils.h"

namespace sad {

class VertexSE2 : public g2o::BaseVertex<3, SE2> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    void setToOriginImpl() override { _estimate = SE2(); }
    void oplusImpl(const double* update) override {
        _estimate.translation()[0] += update[0];
        _estimate.translation()[1] += update[1];
        _estimate.so2() = _estimate.so2() * SO2::exp(update[2]);
    }

    bool read(std::istream& is) override { return true; }
    bool write(std::ostream& os) const override { return true; }
};

class EdgeSE2LikelihoodFiled : public g2o::BaseUnaryEdge<1, double, VertexSE2> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeSE2LikelihoodFiled(const cv::Mat& field_image, double range, double angle, float resolution = 10.0)
        : field_image_(field_image), range_(range), angle_(angle), resolution_(resolution) {}

    /// 判定此条边是否在field image外面
    bool IsOutSide() {
        VertexSE2* v = (VertexSE2*)_vertices[0];
        SE2 pose = v->estimate();
        Vec2d pw = pose * Vec2d(range_ * std::cos(angle_), range_ * std::sin(angle_));
        Vec2i pf = (pw * resolution_ + Vec2d(field_image_.rows / 2, field_image_.cols / 2)).cast<int>();  // 图像坐标

        if (pf[0] >= image_boarder_ && pf[0] < field_image_.cols - image_boarder_ && pf[1] >= image_boarder_ &&
            pf[1] < field_image_.rows - image_boarder_) {
            return false;
        } else {
            return true;
        }
    }

    void computeError() override {
        VertexSE2* v = (VertexSE2*)_vertices[0];
        SE2 pose = v->estimate();
        Vec2d pw = pose * Vec2d(range_ * std::cos(angle_), range_ * std::sin(angle_));
        Vec2d pf =
            pw * resolution_ + Vec2d(field_image_.rows / 2, field_image_.cols / 2) - Vec2d(0.5, 0.5);  // 图像坐标

        if (pf[0] >= image_boarder_ && pf[0] < field_image_.cols - image_boarder_ && pf[1] >= image_boarder_ &&
            pf[1] < field_image_.rows - image_boarder_) {
            _error[0] = math::GetPixelValue<float>(field_image_, pf[0], pf[1]);
        } else {
            _error[0] = 0;
            setLevel(1);
        }
    }

    void linearizeOplus() override {
        VertexSE2* v = (VertexSE2*)_vertices[0];
        SE2 pose = v->estimate();
        float theta = pose.so2().log();
        Vec2d pw = pose * Vec2d(range_ * std::cos(angle_), range_ * std::sin(angle_));
        Vec2d pf =
            pw * resolution_ + Vec2d(field_image_.rows / 2, field_image_.cols / 2) - Vec2d(0.5, 0.5);  // 图像坐标

        if (pf[0] >= image_boarder_ && pf[0] < field_image_.cols - image_boarder_ && pf[1] >= image_boarder_ &&
            pf[1] < field_image_.rows - image_boarder_) {
            // 图像梯度
            float dx = 0.5 * (math::GetPixelValue<float>(field_image_, pf[0] + 1, pf[1]) -
                              math::GetPixelValue<float>(field_image_, pf[0] - 1, pf[1]));
            float dy = 0.5 * (math::GetPixelValue<float>(field_image_, pf[0], pf[1] + 1) -
                              math::GetPixelValue<float>(field_image_, pf[0], pf[1] - 1));

            _jacobianOplusXi << resolution_ * dx, resolution_ * dy,
                -resolution_ * dx * range_ * std::sin(angle_ + theta) +
                    resolution_ * dy * range_ * std::cos(angle_ + theta);
        } else {
            _jacobianOplusXi.setZero();
            setLevel(1);
        }
    }

    bool read(std::istream& is) override { return true; }
    bool write(std::ostream& os) const override { return true; }

   private:
    const cv::Mat& field_image_;
    double range_ = 0;
    double angle_ = 0;
    float resolution_ = 10.0;
    inline static const int image_boarder_ = 10;
};

/**
 * SE2 pose graph使用
 * error = v1.inv * v2 * meas.inv
 */
class EdgeSE2 : public g2o::BaseBinaryEdge<3, SE2, VertexSE2, VertexSE2> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeSE2() {}

    void computeError() override {
        VertexSE2* v1 = (VertexSE2*)_vertices[0];
        VertexSE2* v2 = (VertexSE2*)_vertices[1];
        _error = (v1->estimate().inverse() * v2->estimate() * measurement().inverse()).log();
    }

    // TODO jacobian

    bool read(std::istream& is) override { return true; }
    bool write(std::ostream& os) const override { return true; }

   private:
};

class EdgeSE2P2P : public g2o::BaseUnaryEdge<2, Vec2d, VertexSE2> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeSE2P2P(const pcl::search::KdTree<Icp2d::Point2d>::Ptr kdtree, const Icp2d::Cloud2d::Ptr target_cloud, double range, double angle) : kdtree_(kdtree),  target_cloud_(target_cloud), range_(range), angle_(angle) {}

    // 判断当前激光点的最近邻集合是否为空，或者最小距离是否大于最大距离阈值
    bool isPointValid() { 
        auto* pose = dynamic_cast<const VertexSE2*>(_vertices[0]);
        theta_ = pose->estimate().so2().log(); // 当前位姿的角度
        // 世界系下点的坐标 p_i^W，极坐标转笛卡尔坐标公式
        pw_ = pose->estimate() * Vec2d(range_ * std::cos(angle_), range_ * std::sin(angle_));

        Icp2d::Point2d pt;
        pt.x = pw_.x();
        pt.y = pw_.y();

        // 在目标点云的KD树中查找一个最近邻，返回该最近邻的索引和距离
        kdtree_->nearestKSearch(pt, 1, nn_idx_, dis_);
        float max_dis2 = 0.01;
        // 判断最近邻集合是否非空，且最小距离是否小于最大距离阈值
        if (nn_idx_.size() > 0 && dis_[0] < max_dis2) {
            // 当前激光点在目标点云中的最近邻点坐标
            qw_ = Vec2d(target_cloud_->points[nn_idx_[0]].x, target_cloud_->points[nn_idx_[0]].y);   
            return true;
        }
        else 
            return false;
    }
    
    // 定义残差
    void computeError() override {
        // 判断最近邻集合是否非空，且最小距离是否小于最大距离阈值
        if (isPointValid()) 
            _error =  pw_ - qw_; 
        else {
            _error = Vec2d(0, 0);
            setLevel(1);
        }
    }

    // 雅可比矩阵的解析形式
    void linearizeOplus() override {
        if (isPointValid()) {
            _jacobianOplusXi <<  1, 0, 0, 1,  // de / dx， de / dy
                                -range_ * std::sin(angle_ + theta_), range_ * std::cos(angle_ + theta_);  //  de / dtheta       
        } else {
            _jacobianOplusXi.setZero();
            setLevel(1);
        }                   
    }

    bool read(std::istream& is) override { return true; }
    bool write(std::ostream& os) const override { return true; }

private:
    double range_ = 0;  // 距离
    double angle_ = 0;  // 角度

    double theta_ = 0;
    Vec2d pw_, qw_;
    const pcl::search::KdTree<Icp2d::Point2d>::Ptr kdtree_; 
    const Icp2d::Cloud2d::Ptr target_cloud_;
    std::vector<int> nn_idx_;    // 最近邻的索引
    std::vector<float> dis_;     // 最近邻的距离
};

class EdgeSE2P2L : public g2o::BaseUnaryEdge<1, double, VertexSE2> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeSE2P2L(const pcl::search::KdTree<Icp2d::Point2d>::Ptr kdtree, const Icp2d::Cloud2d::Ptr target_cloud, double range, double angle) : kdtree_(kdtree),  target_cloud_(target_cloud), range_(range), angle_(angle) {}

    bool getIsLineFitSuccess() { return isLineFitSuccess_; }

    // 直线拟合是否成功
    bool isLineFitValid() { 
        auto* pose = dynamic_cast<const VertexSE2*>(_vertices[0]);
        theta_ = pose->estimate().so2().log(); // 当前位姿的角度
        // 世界系下点的坐标 p_i^W，极坐标转笛卡尔坐标公式
        pw_ = pose->estimate() * Vec2d(range_ * std::cos(angle_), range_ * std::sin(angle_));

        Icp2d::Point2d pt;
        pt.x = pw_.x();
        pt.y = pw_.y();

        // 在目标点云的KD树中查找一个最近邻，返回该最近邻的索引和距离
        kdtree_->nearestKSearch(pt, 5, nn_idx_, dis_);

        std::vector<Vec2d> effective_pts;  // 有效点
        float max_dis = 0.3;
        // 遍历所有五个近邻点
        for (int j = 0; j < nn_idx_.size(); ++j) {
            // 判断每个近邻点的距离是否处于最远阈值距离内
            if (dis_[j] < max_dis) 
                // 若是，该近邻点符合要求，存储到向量中
                effective_pts.emplace_back(Vec2d(target_cloud_->points[nn_idx_[j]].x, target_cloud_->points[nn_idx_[j]].y));
        }
        // 判断有效近邻点是否少于三个
        if (effective_pts.size() < 3) 
            // 若少于3个，则跳过当前激光点
            return false;

        
        // 利用当前点附近的几个有效近邻点，基于SVD奇异值分解，拟合出ax+by+c=0 中的最小直线系数 a,b,c，对应公式（6.11）
        if (math::FitLine2D(effective_pts, line_coeffs_)) {
            isLineFitSuccess_ = true;
            return isLineFitSuccess_;
        } else {
            isLineFitSuccess_ = false;
            return isLineFitSuccess_;
        }
    }
    
    // 定义残差
    void computeError() override {
        // 判断最近邻集合是否非空，且最小距离是否小于最大距离阈值
        if (isLineFitValid()) 
            _error[0] = line_coeffs_[0] * pw_[0] + line_coeffs_[1] * pw_[1] + line_coeffs_[2];
        else {
            _error[0] = 0.0;
            setLevel(1);
        }
    }

    // 雅可比矩阵的解析形式
    void linearizeOplus() override {
        if (isLineFitSuccess_) {
            _jacobianOplusXi << line_coeffs_[0], 
                                line_coeffs_[1], 
                                - line_coeffs_[0] * range_ * std::sin(angle_ + theta_) 
                                + line_coeffs_[1] * range_ * std::cos(angle_ + theta_);        
        } else {
            _jacobianOplusXi.setZero();
            setLevel(1);
        }                   
    }

    bool read(std::istream& is) override { return true; }
    bool write(std::ostream& os) const override { return true; }

private:
    double range_ = 0;  // 距离
    double angle_ = 0;  // 角度

    // 【新增】
    double theta_ = 0;
    Vec2d pw_, qw_;
    const pcl::search::KdTree<Icp2d::Point2d>::Ptr kdtree_; 
    const Icp2d::Cloud2d::Ptr target_cloud_;
    std::vector<int> nn_idx_;    // 最近邻的索引
    std::vector<float> dis_;     // 最近邻的距离

    Vec3d line_coeffs_;  // 拟合直线，组装J、H和误差

    bool isLineFitSuccess_ = false;
};

}  // namespace sad

#endif  // SLAM_IN_AUTO_DRIVING_G2O_TYPES_H
