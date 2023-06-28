#include <iostream>  
#include <vector>  
#include <chrono>  
#include "rclcpp/rclcpp.hpp"  
#include "adas_avp_msgs/msg/parking_spots.hpp"  
#include "geometry_msgs/msg/point32.hpp"  
  
using namespace std::chrono_literals;  
  
struct ParkingSpot {  
  geometry_msgs::msg::Point32 position;  
  std::vector<geometry_msgs::msg::Point32> observed_values;  
};  
  
class PSManagerNode : public rclcpp::Node  
{  
public:  
  PSManagerNode() : Node("ps_manager")  
  {  
    // 订阅ParkingSpots消息  
    subscriber_ = create_subscription<adas_avp_msgs::msg::ParkingSpots>(  
        "parking_spots_topic", 10, std::bind(&PSManagerNode::parkingSpotsCallback, this, std::placeholders::_1));  
      
    // 初始化Pseudo_Gt和Observed_Values  
    pseudo_gt_ = {};  
    observed_values_ = {};  
  }  
  
private:  
  void parkingSpotsCallback(const adas_avp_msgs::msg::ParkingSpots::SharedPtr msg)  
  {  
    // 读取时间戳  
    auto timestamp = msg->header.stamp;  
    auto timestamp_sec = timestamp.sec;  
    auto timestamp_nanosec = timestamp.nanosec;  
  
    // 输出时间戳信息  
    std::cout << "Received timestamp: " << timestamp_sec << "." << timestamp_nanosec << std::endl;  
  
    // 读取parking_spots变量  
    std::vector<adas_avp_msgs::msg::ParkingSpot> parkingSpots = msg->parking_spots;  
  
    // 遍历每个观测到的车位  
    for (const auto& spot : parkingSpots)  
    {  
      // 遍历Pseudo_Gt，查找匹配车位  
      bool found_match = false;  
      for (auto& pseudo_gt_spot : pseudo_gt_)  
      {  
        // 计算IOU并进行匹配判断  
        //FIXME: 这里成员变量还得想想怎么改
        double iou = computeIOU(pseudo_gt_spot.position, spot.polygon.points);  
        if (iou > 0.5)  // 设定阈值为0.5  
        {  
          pseudo_gt_spot.observed_values.push_back(spot.polygon.points);  
          found_match = true;  
          break;  
        }  
      }  
  
      // 如果没有匹配的车位，创建一个新车位并添加到Pseudo_Gt  
      if (!found_match)  
      {  
        ParkingSpot new_spot;  
        new_spot.position = spot.polygon.points;  
        new_spot.observed_values.push_back(spot.polygon.points);  
        pseudo_gt_.push_back(new_spot);  
      }  
    }  
  }  
    
  // 计算两个矩形之间的IOU  
  double computeIOU(const geometry_msgs::msg::Point32& rect1, const geometry_msgs::msg::Point32& rect2)  
  {  
    // 计算矩形的面积  
    double rect1_area = computeRectArea(rect1);  
    double rect2_area = computeRectArea(rect2);  
  
    // 计算相交部分的矩形  
    double intersection_area = computeIntersectionArea(rect1, rect2);  
  
    // 计算IOU  
    double iou = intersection_area / (rect1_area + rect2_area - intersection_area);  
  
    return iou;  
  }  
  
  // 计算矩形的面积  
  double computeRectArea(const geometry_msgs::msg::Point32& rect)  
  {  
    double length = std::abs(rect.x - rect.z);  
    double width = std::abs(rect.y - rect.w);  
    return length * width;  
  }  
  
  // 计算相交部分的矩形的面积  
  double computeIntersectionArea(const geometry_msgs::msg::Point32& rect1, const geometry_msgs::msg::Point32& rect2)  
  {  
    double x1 = std::max(rect1.x, rect2.x);  
    double y1 = std::max(rect1.y, rect2.y);  
    double x2 = std::min(rect1.z, rect2.z);  
    double y2 = std::min(rect1.w, rect2.w);  
    
    if (x2 < x1 || y2 < y1)  
      return 0.0;  
    
    double intersection_area = std::abs(x2 - x1) * std::abs(y2 - y1);  
    return intersection_area;  
  }  
  
  std::vector<ParkingSpot> pseudo_gt_;  
  std::vector<ParkingSpot> observed_values_;  
  rclcpp::Subscription<adas_avp_msgs::msg::ParkingSpots>::SharedPtr subscriber_;  
};  
  
int main(int argc, char* argv[])  
{  
  rclcpp::init(argc, argv);  
  
  auto node = std::make_shared<PSManagerNode>();  
  
  rclcpp::spin(node);  
  rclcpp::shutdown();  
  
  return 0;  
}  