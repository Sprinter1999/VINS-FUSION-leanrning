import numpy as np

# 将四元数转换为旋转矩阵
def quaternion_to_rotation_matrix(quaternion):
    x, y, z, w = quaternion
    rotation_matrix = np.array([
        [1-2*y**2-2*z**2, 2*x*y-2*w*z, 2*x*z+2*w*y],
        [2*x*y+2*w*z, 1-2*x**2-2*z**2, 2*y*z-2*w*x],
        [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x**2-2*y**2]
    ])
    return rotation_matrix

# 将四元数和位置转换为位姿矩阵,位姿矩阵后面可以用来算车位的世界坐标
def pose_from_quaternion_and_position(quaternion, position):
    rotation_matrix = quaternion_to_rotation_matrix(quaternion)
    pose_matrix = np.eye(4)
    pose_matrix[:3, :3] = rotation_matrix
    pose_matrix[:3, 3] = position
    return pose_matrix

# 示例
if __name__ == '__main__':
    quaternion = np.array([0.5, 0.5, 0.5, 0.5])
    position = np.array([1.0, 2.0, 3.0])
    pose_matrix = pose_from_quaternion_and_position(quaternion, position)
    print("Pose matrix:\n", pose_matrix)