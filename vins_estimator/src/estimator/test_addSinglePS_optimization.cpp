#include <thread>
#include <mutex>
#include <std_msgs/Header.h>
#include <std_msgs/Float32.h>
#include <ceres/ceres.h>
#include <unordered_map>
#include <queue>
#include <opencv2/core/eigen.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

#include <iostream>
#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>
#include <chrono>

using namespace std;
using namespace Eigen;

//FIXME: 多个车位需要整理成一个double数组
/*计算单个车位PS_loss的残差，单个车位会被多次观测*/
struct CURVE_FITTING_COST{
    CURVE_FITTING_COST(double x,double y, double z):_x(x),_y(y),_z(z){}

    //残差的计算
    template<typename T>
    bool operator()(
        const T *const abc,//模型参数，有3维，为待优化变量组
        T* residual) const{
            //y-exp(ax^2+bx+c)
            residual[0]=T(_y)-ceres::exp(abc[0]*T(_x)*T(_x)+abc[1]*T(_x)+abc[2]);
            return true;
        }

    const double _x,_y;//x，y数据
};

int main(int argc ,char** argv)
{
    /*第一部分，生成观测数据xi,yi*/
	double ar = 1.0, br = 2.0, cr = 1.0;//真实参数值


	double ae = 2.0, be = -1.0, ce = 5.0;//估计参数值,进行初始化

	int N = 100;//数据点
	double w_sigma = 1.0;//噪声的sigma值
	cv::RNG rng;//opencv随机数产生器
	vector<double> x_data, y_data;//数据
	for (int i = 0; i < N; i++)
	{
		double x = i / 100.0;
		x_data.push_back(x);
		y_data.push_back(exp(ar*x*x + br * x + cr) + rng.gaussian(w_sigma*w_sigma));//加上高斯噪声
	}

    /*第二部分，定义残差块*/
    double abc[3]={ae,be,ce};//定义待优化的参数，并赋初值
    ceres::Problem problem;//构建最小二乘问题
    for (int i = 0; i < N; i++)
    {
        //向问题中添加误差项，每一个误差项是一个ResidualBlock
        //每个ResidualBlock由三部分组成：误差项、核函数、待估计参数
        problem.AddResidualBlock(
            //使用自动求导，模板参数：误差类型、输出维度、输入维度，维数要与前面打struct一致
            new ceres::AutoDiffCostFunction<CURVE_FITTING_COST,1,3>(new CURVE_FITTING_COST(x_data[i],y_data[i])),
            nullptr,//核函数，这里不用
            abc//待估参数
        );
    }

    /*第三部分，配置优化器*/
    ceres::Solver::Options options;
    options.linear_solver_type=ceres::DENSE_NORMAL_CHOLESKY;//增量方程如何求解
    options.minimizer_progress_to_stdout=true;//输出到cout
    ceres::Solver::Summary summary;//优化信息

    chrono::steady_clock::time_point t1=chrono::steady_clock::now();//记录开始时间
    ceres::Solve(options,&problem,&summary);//开始优化
    chrono::steady_clock::time_point t2=chrono::steady_clock::now();//记录结束时间
    chrono::duration<double> time_used=chrono::duration_cast<chrono::duration<double>>(t2-t1);//计算用时
    cout<<"solve time cost="<<time_used.count()<<" seconds"<<endl;

    /*第四部分  输出结果*/
    cout<<summary.BriefReport()<<endl;
    cout<<"estimated a,b,c=";
    for (auto a:abc)cout<<a<<" ";
    cout<<endl;
    
    
    return 0;
}
