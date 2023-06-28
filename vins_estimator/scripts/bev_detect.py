# Description: This module served to detect the parking spot in the BEV image and used the current pose to convey the spot corner points to the PS_manager Node.
# Developed by xuefeng jiang on 2023.6.20
# FIXME: should be suitably combined with the xauto codes, some message interface should be added and compiled
# TODO: please help to consider the following issue:
# TODO: I think BEV image should be timestamped with the related around-view camera images, and the timestamp should be used to retrieve the corresponding odometry
# Subscribed topics:
                    # -  C++ Node published nav_msgs::Odometry odometry, we temporarily named it as /vins_estimator/odometry
                    # -  BEV image
 

#!/usr/bin/env python3
import os
from pathlib import Path
from turtle import right
import numpy as np
import uuid
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import Image, CameraInfo
from adas_avp_msgs.msg import Image as XImage
from nav_msgs.msg import Odometry 
from geometry_msgs.msg import Point32, Polygon, Point, Quaternion
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
from tf2_ros.transform_listener import TransformListener
from tf2_ros.buffer import Buffer
import io
import cv2
import json
from cv_bridge import CvBridge, CvBridgeError

#FIXME: add psdet yolov5 to this folder
from adas_avp_msgs.msg import ParkingSpots, ParkingSpot
from psdet.yolov5_vps.models.vps_direction import PsDetect
from psdet.yolov5_vps.utils.vps_utils import compute_four_points_direction
import copy



# fixed parameters
bev_height = 600
bev_width = 600
scale = 53
baselinkfp_to_vechiclefp = -1.369
uv_to_vehiclefp2 = np.zeros((4, 4), dtype=np.float64)
uv_to_vehiclefp2[0, 3] = bev_height / 2
uv_to_vehiclefp2[1, 3] = bev_width / 2
uv_to_vehiclefp2[2, 3] = 0
uv_to_vehiclefp2[3, 3] = 1
uv_to_vehiclefp2[:3, :3] = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]],
                                    dtype=np.float64)
baselinkfp_to_vehiclefp = np.zeros((4, 4), dtype=np.float64)
baselinkfp_to_vehiclefp[0, 3] = baselinkfp_to_vechiclefp
baselinkfp_to_vehiclefp[1, 3] = 0
baselinkfp_to_vehiclefp[2, 3] = 0
baselinkfp_to_vehiclefp[3, 3] = 1
baselinkfp_to_vehiclefp[:3, :3] = np.identity(3, dtype=np.float64)
vehiclefp_to_baselinkfp = np.linalg.inv(baselinkfp_to_vehiclefp)


def bev_points_to_baselinkfp(pts):
    pts = np.array([[i[0], i[1], 0, 1] for i in pts],
                   dtype=np.float64).reshape(-1, 4)
    XYZ = (uv_to_vehiclefp2 @ pts.T).T[:, :3] / scale
    XYZ = np.concatenate(
        [XYZ, np.ones((XYZ.shape[0], 1), dtype=np.float64)], axis=-1)
    XYZ = (vehiclefp_to_baselinkfp @ XYZ.T).T[:, :3]
    return XYZ


def to_geometry_point_msg(pt):
    return Point32(x=float(pt[0]), y=float(pt[1]))

def to_area_msg(detection, header, nearest_pose):
    point1 = detection[0]
    point2 = detection[1]
    angle = detection[2]
    direction = detection[3]
    conf = detection[4]
    label_vacant = detection[5]
    pts = compute_four_points_direction(angle, point1, point2, direction)
    pts = bev_points_to_baselinkfp(pts)

    #FIXME: retrieve the current pose from the odometry topic
    cur_odom = nearest_pose
    pts_baslinkfp_to_worldfp = [compute_ps_in_worldframe(cur_odom,each) for each in pts]
    # point3_org = copy.copy(pts[2])
    # point4_org = copy.copy(pts[3])
    parking_spot = ParkingSpot()
    parking_spot.free = True if label_vacant == 0 else False
    area = Polygon()
    for i in [0, 3, 2, 1]:
        area.points.append(to_geometry_point_msg(pts_baslinkfp_to_worldfp[i]))
    parking_spot.header = header
    parking_spot.polygon = area
    return parking_spot


class PSDetNode(Node):
    def __init__(self):
        super().__init__('psdet_node') # 指定节点名称
        # qos_profile = rclpy.qos.QoSProfile(depth=10)
        # qos_profile.reliability = rclpy.qos.QoSReliabilityPolicy.RELIABLE
        # qos_profile.history = rclpy.qos.QoSHistoryPolicy.KEEP_ALL
        # qos_profile.durability = rclpy.qos.QoSDurabilityPolicy.VOLATILE
        
        # 相机频率会比较低
        self.image_sub = self.create_subscription(
            XImage,
            'bev_image',
            self.cb,
            40)
        
        self.camera_pose_sub = self.create_subscription(
            Odometry,
            'vins_estimator/camera_pose',
            self.cb2,
            300)
        
        self.observed_camera_pose_list = []
        self.observed_camera_pose_timestamp_list = []
            

        self.bridge = CvBridge()
        self.last_seq_id = None

        data = self.declare_parameter('data').value
        weights_path_yolo = self.declare_parameter('weights').value
        self.conf_thres = self.declare_parameter('conf_thres', 0.7).value
        self.nms_thres = self.declare_parameter('nms_thres', 0.5).value
        self.debug = self.declare_parameter('debug', None).value
        device = self.declare_parameter('device', 'cpu').value
        img_size = 576
        num_class = 13

        # device = "cpu"
        self.ps_detect = PsDetect(weights_path_yolo, img_size, device, data=data, half=False, nc=num_class)
        if self.debug:
            self.debug_publisher = self.create_publisher(
                Image, 'output_debug', 10)
        else:
            self.debug_publisher = None

        self.publisher = self.create_publisher(ParkingSpots, 'output', 10)
        # self.tf_buffer = Buffer()
        # self.tf_listener = TransformListener(self.tf_buffer, self)


    def cb2(self, msg):
        # 接收到新的pose信息，从header中获取时间戳，并将时间戳和pose信息存入对应列表
        # pose列表主要是缓冲信息，用于后续和bev图像的时间戳进行邻近匹配，进而去计算车位的世界坐标
        self.observed_camera_pose_list.append(msg.pose.pose)
        self.observed_camera_pose_timestamp_list.append(msg.header.stamp)
        # 如果列表中的pose信息数量大于200，则删除最早的一个pose信息，否则要处理的pose信息太多，会导致计算量过大，这个可能需要调一下
        if len(self.observed_camera_pose_list) > 200:
            self.observed_camera_pose_list.pop(0)
            self.observed_camera_pose_timestamp_list.pop(0)

        return

    #TODO: 目前考虑地比较理想，认为一定能找到一个较为合理的邻近值，实际上可能会有bad case
    def find_nearest_timestamp(self, bev_timestamp):
        timestamp_list = self.observed_camera_pose_timestamp_list
        nearest_timestamp = None
        nearest_index = None
        min_diff = float('inf')

        for i, timestamp in enumerate(timestamp_list):
            diff = abs(timestamp - bev_timestamp)
            if diff < min_diff:
                min_diff = diff
                nearest_timestamp = timestamp
                nearest_index = i

        return nearest_timestamp, nearest_index

    def cb(self, msg):
        cv_img = self.bridge.imgmsg_to_cv2(msg.base)
        img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

        seq_id = msg.extend.seq_id
        if self.last_seq_id is None:
            self.last_seq_id = seq_id
        else:
            if self.last_seq_id >= seq_id:
                return
            self.last_seq_id = seq_id


        stamp = msg.header.stamp

        # 根据bev图像的时间戳，找到最近的pose信息
        nearest_timestamp, nearest_index = self.find_nearest_timestamp(stamp)
        nearest_pose = self.observed_camera_pose_list[nearest_index]

        frame_id = msg.header.frame_id
        detections = self.ps_detect.detect_ps(img, self.conf_thres, self.nms_thres)
        # Create result 函数会完成车位的世界坐标计算
        result = self.create_results(detections, msg.header,nearest_pose)

        # TODO: psdet的精度大概有多少之后可以找yixiong问一下
        # if self.debug_publisher is not None:
        #     if len(detections) != 0:
        #         for detection in detections:
        #             point1 = detection[0]
        #             point2 = detection[1]
        #             angle = detection[2]
        #             direction = detection[3]
        #             conf = detection[4]
        #             vacant = detection[5]

        #             pts = compute_four_points_direction(angle, point1, point2, direction)

        #             # pts[0] and pts[1] are marking points of the entrance
        #             pts_show = np.array([pts[0], pts[1], pts[2], pts[3]], np.int32)
        #             # vacant = 0 means the slot is empty
                    
        #             if vacant == 0:
        #                 color = (0, 0, 255)#(r,g,b)
        #             else:
        #                 color = (255, 0, 0)

        #             cv2.polylines(cv_img, [pts_show], True, color, 2)
        #             # cv2.imwrite(os.path.join('/home/adas/hanl/xautoproj/tmp', f"{seq_id}.jpg"), img[:,:,::-1])
        #     self.debug_publisher.publish(
        #         self.bridge.cv2_to_imgmsg(cv_img, encoding='bgr8'))

        self.publisher.publish(result)

    def create_results(self, detections, header, nearest_pose):
        result = ParkingSpots()
        new_header = Header()
        new_header.stamp = header.stamp
        new_header.frame_id = 'base_link'
        result.header = new_header
        for detection in detections:
            result.parking_spots.append(to_area_msg(detection, new_header, nearest_pose))
        return result

# 将四元数转换为旋转矩阵
def quaternion_to_rotation_matrix(x, y, z, w):
    rotation_matrix = np.array([
        [1-2*y**2-2*z**2, 2*x*y-2*w*z, 2*x*z+2*w*y],
        [2*x*y+2*w*z, 1-2*x**2-2*z**2, 2*y*z-2*w*x],
        [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x**2-2*y**2]
    ])
    return rotation_matrix

# 将四元数和位置转换为位姿矩阵,位姿矩阵后面可以用来算车位的世界坐标
# def pose_from_quaternion_and_position(quaternion, position):
#     rotation_matrix = quaternion_to_rotation_matrix(quaternion)
#     pose_matrix = np.eye(4)
#     pose_matrix[:3, :3] = rotation_matrix
#     pose_matrix[:3, 3] = position
#     return pose_matrix


def compute_ps_in_worldframe(odom, feature):
    # 获取自车在世界坐标系下的位姿
    tx = odom.pose.pose.position.x
    ty = odom.pose.pose.position.y
    tz = odom.pose.pose.position.z
    qx = odom.pose.pose.orientation.x
    qy = odom.pose.pose.orientation.y
    qz = odom.pose.pose.orientation.z
    qw = odom.pose.pose.orientation.w

    # 获取特征点在自车坐标系下的坐标
    x = feature.x
    y = feature.y
    z = feature.z

    # 计算自车在世界坐标系下的旋转矩阵和平移矩阵
    R = quaternion_to_rotation_matrix(qx, qy, qz, qw)
    T = np.array([[1, 0, 0, tx],
                  [0, 1, 0, ty],
                  [0, 0, 1, tz],
                  [0, 0, 0, 1]])
    RT = np.dot(R, T)

    # 将特征点坐标转换为齐次坐标
    X = x
    Y = y
    Z = z
    W = 1
    feature_hom = np.array([[X], [Y], [Z], [W]])

    # 将特征点坐标从自车坐标系转换到世界坐标系
    feature_w = np.dot(RT, feature_hom)
    x_w = feature_w[0][0] / feature_w[3][0]
    y_w = feature_w[1][0] / feature_w[3][0]
    z_w = feature_w[2][0] / feature_w[3][0]

    # 输出特征点在世界坐标系下的坐标
    print("Feature point (world): ({}, {}, {})".format(x_w, y_w, z_w))



def main():
    rclpy.init()
    node = PSDetNode()
    rclpy.spin(node)

if __name__ == '__main__':
    main()

