#include <rclcpp/rclcpp.hpp>
#include <ceres/ceres.h>
#include <back_opt_msgs/srv/optimize.hpp>
#include <geometry_msgs/msg/vector3.hpp>

using namespace std::placeholders;

// 定义一个 ROS2 节点类
class BackOptNode : public rclcpp::Node
{
public:
    BackOptNode()
        : Node("back_opt")
    {
        // 创建一个 ROS2 Service，等待 PS_manager 节点调用
        service_ = create_service<back_opt_msgs::srv::Optimize>(
            "back_optimize", std::bind(&BackOptNode::optimize_callback, this, _1, _2));
    }

private:
    // Service 回调函数
    void optimize_callback(const std::shared_ptr<rmw_request_id_t> request_header,
                           const std::shared_ptr<back_opt_msgs::srv::Optimize::Request> request,
                           std::shared_ptr<back_opt_msgs::srv::Optimize::Response> response)
    {
        // 从请求消息中获取待优化变量和观测值
        const auto& pseudo_gt = request->pseudo_gt;
        const auto& observed_values = request->observed_values;

        // 创建 Ceres 残差函数对象
        auto cost_function = ceres::CostFunction::Create(
            new ceres::AutoDiffCostFunction<Residual, 3, 3>(new Residual(observed_values)));

        // 创建 Ceres 优化问题对象
        ceres::Problem problem;
        problem.AddResidualBlock(cost_function, nullptr, pseudo_gt.data());

        // 配置 Ceres 优化器参数
        ceres::Solver::Options options;
        options.max_num_iterations = 100;
        options.linear_solver_type = ceres::DENSE_QR;
        options.minimizer_progress_to_stdout = true;

        // 创建 Ceres 优化器对象，并开始优化
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        // 将优化结果写入响应消息
        response->optimized_pseudo_gt.x = pseudo_gt[0];
        response->optimized_pseudo_gt.y = pseudo_gt[1];
        response->optimized_pseudo_gt.z = pseudo_gt[2];
    }

    // 定义残差函数类
    class Residual
    {
    public:
        Residual(const std::vector<geometry_msgs::msg::Vector3>& observed_values)
            : observed_values_(observed_values)
        {
        }

        template <typename T>
        bool operator()(const T* const pseudo_gt, T* residual) const
        {
            for (size_t i = 0; i < observed_values_.size(); i++)
            {
                const auto& obs = observed_values_[i];
                T diff[3] = {pseudo_gt[0] - T(obs.x), pseudo_gt[1] - T(obs.y), pseudo_gt[2] - T(obs.z)};
                residual[i * 3 + 0] = diff[0];
                residual[i * 3 + 1] = diff[1];
                residual[i * 3 + 2] = diff[2];
            }
            return true;
        }

    private:
        const std::vector<geometry_msgs::msg::Vector3>& observed_values_;
    };

    // ROS2 Service 对象
    rclcpp::Service<back_opt_msgs::srv::Optimize>::SharedPtr service_;
};

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<BackOptNode>());
    rclcpp::shutdown();
    return 0;
}