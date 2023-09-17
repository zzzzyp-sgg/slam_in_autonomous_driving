//
// Created by xiang on 2022/3/15.
//

#include "ch6/icp_2d.h"
#include "common/math_utils.h"
#include "ch6/g2o_types.h"

#include <glog/logging.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/search/impl/kdtree.hpp>

#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

namespace sad {

bool Icp2d::AlignGaussNewton(SE2& init_pose) {
    int iterations = 10;
    double cost = 0, lastCost = 0;
    SE2 current_pose = init_pose;
    const float max_dis2 = 0.01;    // 最近邻时的最远距离（平方）
    const int min_effect_pts = 20;  // 最小有效点数

    for (int iter = 0; iter < iterations; ++iter) {
        Mat3d H = Mat3d::Zero();
        Vec3d b = Vec3d::Zero();
        cost = 0;

        int effective_num = 0;  // 有效点数

        // 遍历source
        // TODO source应该是指当前scan
        for (size_t i = 0; i < source_scan_->ranges.size(); ++i) {
            float r = source_scan_->ranges[i];
            if (r < source_scan_->range_min || r > source_scan_->range_max) {
                continue;
            }

            float angle = source_scan_->angle_min + i * source_scan_->angle_increment;
            float theta = current_pose.so2().log();
            // p_W_i = T_W_B * p_B_i
            Vec2d pw = current_pose * Vec2d(r * std::cos(angle), r * std::sin(angle));
            Point2d pt;
            pt.x = pw.x();
            pt.y = pw.y();

            // 最近邻
            std::vector<int> nn_idx;
            std::vector<float> dis;
            kdtree_.nearestKSearch(pt, 1, nn_idx, dis); // 这里应该是找到最近邻的一个点

            if (nn_idx.size() > 0 && dis[0] < max_dis2) {
                effective_num++;
                Mat32d J;
                // Jacobian (6.8)
                J << 1, 0, 0, 1, -r * std::sin(angle + theta), r * std::cos(angle + theta);
                H += J * J.transpose();

                // residual
                Vec2d e(pt.x - target_cloud_->points[nn_idx[0]].x, pt.y - target_cloud_->points[nn_idx[0]].y);
                b += -J * e;

                cost += e.dot(e);
            }
        }

        if (effective_num < min_effect_pts) {
            return false;
        }

        // solve for dx
        Vec3d dx = H.ldlt().solve(b);
        if (isnan(dx[0])) {
            break;
        }

        cost /= effective_num;
        if (iter > 0 && cost >= lastCost) {
            break;
        }

        LOG(INFO) << "iter " << iter << " cost = " << cost << ", effect num: " << effective_num;

        current_pose.translation() += dx.head<2>();
        current_pose.so2() = current_pose.so2() * SO2::exp(dx[2]);
        lastCost = cost;
    }

    init_pose = current_pose;
    LOG(INFO) << "estimated pose: " << current_pose.translation().transpose()
              << ", theta: " << current_pose.so2().log();

    return true;
}

bool Icp2d::AlignG2O(SE2& init_pose) {
    int iterations = 10;          // 迭代次数
    double rk_delta = 0.8;
    float max_dis2 = 0.01;        // 最近邻时的最远距离（平方）
    int min_effect_pts = 20;      // 最小有效点数

    using BlockSolverType = g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>>;
    using LinearSolverType = g2o::LinearSolverCholmod<BlockSolverType::PoseMatrixType>;
    auto* solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    
    auto* v = new VertexSE2();      // 新建SE2位姿顶点
    v->setId(0);                    // 设置顶点的id
    v->setEstimate(init_pose);   // 设置顶点的估计值为初始位姿
    optimizer.addVertex(v);         // 将顶点添加到优化器中
    int effective_num = 0;  // 有效点数
    // 遍历源始点云
    for (size_t i = 0; i < source_scan_->ranges.size(); ++i) {
        double range = source_scan_->ranges[i]; // 源始点云的距离
        // 判断每个点的距离是否越界
        if (range < source_scan_->range_min || range > source_scan_->range_max) 
            continue;

        // 根据最小角度和角分辨率计算每个点的角度
        double angle = source_scan_->angle_min + i * source_scan_->angle_increment;
        
        auto *edge = new EdgeSE2P2P(kdtree_g2o_, target_cloud_, range, angle);   

        edge->setVertex(0, v);                  // 设置边的第一个顶点为SE2位姿顶点
        if (edge->isPointValid()){
            effective_num++; 
            edge->setInformation(Mat2d::Identity());// 观测为2维点坐标，信息矩阵需设为2x2单位矩阵
            auto rk = new g2o::RobustKernelHuber;   // Huber鲁棒核函数
            rk->setDelta(rk_delta);                 // 设置阈值
            edge->setRobustKernel(rk);              // 为边设置鲁棒核函数    
            optimizer.addEdge(edge);                // 将约束边添加到优化器中
        }
    }

    // 判断有效激光点数是否少于最小有效点数阈值
    if (effective_num < min_effect_pts) 
        return false;

    optimizer.setVerbose(false);        // 不输出优化过程
    optimizer.initializeOptimization(); // 初始化优化器
    optimizer.optimize(10);              // g2o内部仅非线性优化求解一次

    // 取出优化后的SE2位姿，更新当前位姿，用于下一次迭代
    init_pose = v->estimate();
    LOG(INFO) << "estimated pose: " << v->estimate().translation().transpose() << ", theta: " << v->estimate().so2().log();
    return true;
}

bool Icp2d::DegradationDetection(SE2& init_pose) {
        int iterations = 10;
    double cost = 0, lastCost = 0;
    SE2 current_pose = init_pose;
    const float max_dis = 0.3;     // 最近邻时的最远距离
    const int min_effect_pts = 20; // 最小有效点数

    // 遍历source，使用直线拟合查看scan是否退化
    std::vector<Vec2d> ab_coeffs;
    for (size_t i = 0; i < source_scan_->ranges.size(); ++i)
    {
        float r = source_scan_->ranges[i];
        if (r < source_scan_->range_min || r > source_scan_->range_max)
        {
            continue;
        }

        float angle = source_scan_->angle_min + i * source_scan_->angle_increment;
        float theta = current_pose.so2().log();
        Vec2d pw = current_pose * Vec2d(r * std::cos(angle), r * std::sin(angle));
        Point2d pt;
        pt.x = pw.x();
        pt.y = pw.y();

        // 查找5个最近邻
        std::vector<int> nn_idx;
        std::vector<float> dis;
        kdtree_.nearestKSearch(pt, 5, nn_idx, dis);

        std::vector<Vec2d> effective_pts; // 有效点
        for (int j = 0; j < nn_idx.size(); ++j)
        {
            if (dis[j] < max_dis)
            {
                effective_pts.emplace_back(
                    Vec2d(target_cloud_->points[nn_idx[j]].x, target_cloud_->points[nn_idx[j]].y));
            }
        }

        if (effective_pts.size() < 3)
        {
            continue;
        }

        // 拟合直线
        Vec3d line_coeffs;
        if (math::FitLine2D(effective_pts, line_coeffs))
        {
            ab_coeffs.emplace_back(Vec2d(line_coeffs[0], line_coeffs[1]));
        }
    }
    // 进行SVD分解，利用特征值大小判断scan是否退化
    Eigen::MatrixXd A(ab_coeffs.size(), 2);
    for (size_t i = 0; i < ab_coeffs.size(); ++i)
    {
        A.block(i, 0, 1, 2) << ab_coeffs[i][0], ab_coeffs[i][1];
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A * A.transpose(), Eigen::ComputeFullU | Eigen::ComputeFullV);
    double sum_sing_val = 0.0;
    // std::cout << "A矩阵奇异值为： \n";
    for (size_t i = 0; i < svd.singularValues().size(); ++i)
    {
        if (svd.singularValues()[i] > 0.05)
        {
            std::cout << svd.singularValues()[i] << std::endl;
        }
        sum_sing_val += svd.singularValues()[i];
    }
    if (svd.singularValues()[0] / sum_sing_val >= 0.95)
    {
        std::cout << "退化的scan!!! 最大奇异值占比" << svd.singularValues()[0] / sum_sing_val << std::endl;
        return true;
    }
    else
    {
        std::cout << "并未退化，最大奇异值占比" << svd.singularValues()[0] / sum_sing_val << std::endl;
        return true;
    }
}

bool Icp2d::AlignGaussNewtonPoint2Plane(SE2& init_pose) {
    int iterations = 10;
    double cost = 0, lastCost = 0;
    SE2 current_pose = init_pose;
    const float max_dis = 0.3;      // 最近邻时的最远距离
    const int min_effect_pts = 20;  // 最小有效点数

    for (int iter = 0; iter < iterations; ++iter) {
        Mat3d H = Mat3d::Zero();
        Vec3d b = Vec3d::Zero();
        cost = 0;

        int effective_num = 0;  // 有效点数

        // 遍历source
        for (size_t i = 0; i < source_scan_->ranges.size(); ++i) {
            float r = source_scan_->ranges[i];
            if (r < source_scan_->range_min || r > source_scan_->range_max) {
                continue;
            }

            float angle = source_scan_->angle_min + i * source_scan_->angle_increment;
            float theta = current_pose.so2().log();
            Vec2d pw = current_pose * Vec2d(r * std::cos(angle), r * std::sin(angle));
            Point2d pt;
            pt.x = pw.x();
            pt.y = pw.y();

            // 查找5个最近邻
            std::vector<int> nn_idx;
            std::vector<float> dis;
            kdtree_.nearestKSearch(pt, 5, nn_idx, dis); // 点到线，所以要查找大于1个最近邻

            std::vector<Vec2d> effective_pts;  // 有效点
            for (int j = 0; j < nn_idx.size(); ++j) {
                if (dis[j] < max_dis) {
                    effective_pts.emplace_back(
                        Vec2d(target_cloud_->points[nn_idx[j]].x, target_cloud_->points[nn_idx[j]].y));
                }
            }

            if (effective_pts.size() < 3) { // 最近邻的有效点的个数不能小于3
                continue;
            }

            // 拟合直线，组装J、H和误差
            Vec3d line_coeffs;
            if (math::FitLine2D(effective_pts, line_coeffs)) {
                effective_num++;
                Vec3d J;
                // (6.16) && (6.17)
                J << line_coeffs[0], line_coeffs[1],
                    -line_coeffs[0] * r * std::sin(angle + theta) + line_coeffs[1] * r * std::cos(angle + theta);
                H += J * J.transpose();

                double e = line_coeffs[0] * pw[0] + line_coeffs[1] * pw[1] + line_coeffs[2];
                b += -J * e;

                cost += e * e;
            }
        }

        if (effective_num < min_effect_pts) {
            return false;
        }

        // solve for dx
        Vec3d dx = H.ldlt().solve(b);
        if (isnan(dx[0])) {
            break;
        }

        cost /= effective_num;
        if (iter > 0 && cost >= lastCost) {
            break;
        }

        LOG(INFO) << "iter " << iter << " cost = " << cost << ", effect num: " << effective_num;

        current_pose.translation() += dx.head<2>();
        current_pose.so2() = current_pose.so2() * SO2::exp(dx[2]);
        lastCost = cost;
    }

    init_pose = current_pose;
    LOG(INFO) << "estimated pose: " << current_pose.translation().transpose()
              << ", theta: " << current_pose.so2().log();

    return true;
}

bool Icp2d::AlignG2oPoint2Plane(SE2& init_pose) {
    int iterations = 10;        // 迭代次数
    double rk_delta = 0.8;
    float max_dis = 0.3;       // 最近邻时的最远距离（平方）
    int min_effect_pts = 20;    // 最小有效点数
    
    SE2 current_pose = init_pose;   // 当前位姿

    using BlockSolverType = g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>>;
    using LinearSolverType = g2o::LinearSolverCholmod<BlockSolverType::PoseMatrixType>;
    auto* solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    auto* v = new VertexSE2();      // 新建SE2位姿顶点
    v->setId(0);                    // 设置顶点的id
    v->setEstimate(current_pose);   // 设置顶点的估计值为初始位姿
    optimizer.addVertex(v);         // 将顶点添加到优化器中
    int effective_num = 0;          // 有效点数

    // 遍历源始点云
    for (size_t i = 0; i < source_scan_->ranges.size(); ++i) {
        double range = source_scan_->ranges[i]; // 源始点云的距离
        // 判断每个点的距离是否越界
        if (range < source_scan_->range_min || range > source_scan_->range_max) 
            continue;

        // 当前激光点的角度
        double angle = source_scan_->angle_min + i * source_scan_->angle_increment;
        
        auto *edge = new EdgeSE2P2L(kdtree_g2o_, target_cloud_, range, angle);

        edge->setVertex(0, v);                  // 设置边的第一个顶点为SE2位姿顶点
        

        // 利用当前点附近的几个有效近邻点，基于SVD奇异值分解，拟合出ax+by+c=0 中的最小直线系数 a,b,c，对应公式（6.11）
        if (edge->isLineFitValid()) {
            effective_num++; // 有效点数
            edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity());// 观测为2维点坐标，因此信息矩阵需设为2x2单位矩阵
            auto rk = new g2o::RobustKernelHuber;   // Huber鲁棒核函数
            rk->setDelta(rk_delta);                 // 设置阈值
            edge->setRobustKernel(rk);              // 为边设置鲁棒核函数
            optimizer.addEdge(edge);                // 将约束边添加到优化器中
        }
    }

    // 判断有效激光点数是否少于最小有效点数阈值
    if (effective_num < min_effect_pts) 
        return false;

    optimizer.setVerbose(false);        // 不输出优化过程
    optimizer.initializeOptimization(); // 初始化优化器
    optimizer.optimize(10);              // g2o内部仅非线性优化求解一次

    // 取出优化后的SE2位姿，更新当前位姿，用于下一次迭代
    current_pose = v->estimate();
    
    init_pose = current_pose;
    LOG(INFO) << "estimated pose: " << current_pose.translation().transpose() << ", theta: " << current_pose.so2().log();
    // LOG(INFO) << "g2o: estimated pose: " << init_pose.translation().transpose() << ", theta: " << init_pose.so2().log();
    return true;   
}

void Icp2d::BuildTargetKdTree() {
    if (target_scan_ == nullptr) {
        LOG(ERROR) << "target is not set";
        return;
    }

    target_cloud_.reset(new Cloud2d);
    for (size_t i = 0; i < target_scan_->ranges.size(); ++i) {
        if (target_scan_->ranges[i] < target_scan_->range_min || target_scan_->ranges[i] > target_scan_->range_max) {
            continue;
        }

        // theta + rou_i
        double real_angle = target_scan_->angle_min + i * target_scan_->angle_increment;

        Point2d p;
        p.x = target_scan_->ranges[i] * std::cos(real_angle);   // r_i * cos(theta)
        p.y = target_scan_->ranges[i] * std::sin(real_angle);   // r_i * sin(theta)
        target_cloud_->points.push_back(p);
    }

    target_cloud_->width = target_cloud_->points.size();
    target_cloud_->is_dense = false;
    kdtree_.setInputCloud(target_cloud_);

    kdtree_g2o_ = boost::make_shared<pcl::search::KdTree<Point2d>>();
    kdtree_g2o_->setInputCloud(target_cloud_);
}

}  // namespace sad