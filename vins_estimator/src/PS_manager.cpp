#include <ros/ros.h>
#include <std_msgs/Float32MultiArray.h>
#include <Eigen/Core>
#include <vector>
#include <algorithm>

using Eigen::Vector3d;

struct Spot
{
    int id;
    Vector3d Pseudo_Gt;
    std::vector<Vector3d> Observed_Values;
};

class PS_manager
{
public:
    PS_manager() : nh_("~")
    {
        sub_ = nh_.subscribe("detections", 1, &PS_manager::detectionsCallback, this);
    }

    void detectionsCallback(const std_msgs::Float32MultiArray::ConstPtr &msg)
    {
        int num_spots = msg->data.size() / 3;

        for (int i = 0; i < num_spots; i++)
        {
            int offset = i * 3;
            Vector3d observation;
            observation << msg->data[offset], msg->data[offset + 1], msg->data[offset + 2];

            match_observation(observation);
        }
    }

    void match_observation(const Vector3d &obs)
    {
        auto iter = std::find_if(ParkingSpots.begin(), ParkingSpots.end(),
                                 [&](const Spot &spot)
                                 {
                                     Vector3d diff = spot.Pseudo_Gt - obs;
                                     double distance = diff.norm();
                                     return distance < threshold_;
                                 });

        if (iter != ParkingSpots.end())
        {
            iter->Observed_Values.push_back(obs);
        }
        else
        {
            int new_id = ParkingSpots.empty() ? 0 : ParkingSpots.back().id + 1;
            Spot new_spot;
            new_spot.id = new_id;
            new_spot.Pseudo_Gt = obs;
            new_spot.Observed_Values.push_back(obs);
            ParkingSpots.push_back(new_spot);
        }
    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber sub_;
    std::vector<Spot> ParkingSpots;
    double threshold_ = 0.5; //判断是否为同一停车位的阈值，需要调整
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "PS_manager");
    PS_manager node;
    ros::spin();
    return 0;
}