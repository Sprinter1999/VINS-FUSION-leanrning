#include <iostream>
#include "ceres/ceres.h"

// 定义Cost Function
struct QuadraticLSEFunctor
{
    QuadraticLSEFunctor(double x, double y,double z) : x(x), y(y), z(z) {}

    template <typename T>
    bool operator()(const T* const abc, T* residual) const
    {
        residual[0] = (abc[0]-x) *(abc[0]-x) + (abc[1]-y) *(abc[1]-y) + (abc[2]-z) *(abc[2]-z);
        return true;
    }

private:
    const double x;
    const double y;
};

int main(int argc, char** argv)
{
    // 生成一些对于同一车位的观测
    const int num_points = 5;
    double x_data[num_points];
    double y_data[num_points];
    double z_data[num_points];
    
    //读取数值给上述数组

    // 需要趋近的真实车位坐标,用第一次观测初始化
    double x_r = x_data[0];
    double y_r = y_data[0];
    double z_r = z_data[0];


    // 构建Ceres Solver问题
    ceres::Problem problem;

    // 添加Cost Function
    for (int i = 0; i < num_points; ++i)
    {
        ceres::CostFunction* cost_function =
            new ceres::AutoDiffCostFunction<QuadraticFunctor, 1, 3>(
                new QuadraticFunctor(x_data[i], y_data[i]));
        problem.AddResidualBlock(cost_function, NULL, &a, &b, &c);
    }

    // 配置Solver选项
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    options.trust_region_strategy_type = ceres::DOGLEG;

    // 运行Solver求解
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // 输出结果
    std::cout << summary.FullReport() << std::endl;
    std::cout << "a = " << a << ", b = " << b << ", c = " << c << std::endl;

    return 0;
}