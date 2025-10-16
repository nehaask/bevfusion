import numpy as np
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points

from load_data import load_lidar_camera_sample

def get_calibration_matrices(nusc, lidar_token, camera_token):
    """
    Get calibration matrices to transform from LiDAR to camera coordinate frame
    
    inputs:
        nusc: NuScenes instance
        lidar_token: LiDAR sample_data token
        camera_token: Camera sample_data token
    
    Returns:
        cam_intrinsic: Camera intrinsic matrix (3x3)
        lidar_to_cam_transform: Transformation matrix from LiDAR to camera (4x4)
    """
    # geting lidar and camera sample data
    lidar_sample = nusc.get('sample_data', lidar_token)
    camera_sample = nusc.get('sample_data', camera_token)
    
    # getting calibrated sensor records (contain extrinsics and intrinsics)
    lidar_calib = nusc.get('calibrated_sensor', lidar_sample['calibrated_sensor_token'])
    camera_calib = nusc.get('calibrated_sensor', camera_sample['calibrated_sensor_token'])
    
    # camera intrinsic matrix (3x3)
    # maps 3D points in camera frame to 2D pixel coordinates
    cam_intrinsic = np.array(camera_calib['camera_intrinsic'])
    
    print("\nCamera Intrinsic Matrix:")
    print("(Maps 3D camera coords → 2D pixels)")
    print(cam_intrinsic)
    print(f"  fx = {cam_intrinsic[0,0]:.2f} (focal length X)")
    print(f"  fy = {cam_intrinsic[1,1]:.2f} (focal length Y)")
    print(f"  cx = {cam_intrinsic[0,2]:.2f} (principal point X)")
    print(f"  cy = {cam_intrinsic[1,2]:.2f} (principal point Y)")
    
    # getting ego pose (vehicle position in world frame) for both sensors
    lidar_ego_pose = nusc.get('ego_pose', lidar_sample['ego_pose_token'])
    camera_ego_pose = nusc.get('ego_pose', camera_sample['ego_pose_token'])
    
    
    # Build transformation matrices

    # Step 1: LiDAR sensor frame → vehicle frame
    lidar_translation = np.array(lidar_calib['translation'])
    lidar_rotation = np.array(lidar_calib['rotation'])  # quaternion [w, x, y, z] (NuScenes format)
    lidar_to_ego = transform_matrix(lidar_translation, lidar_rotation)
    
    # Step 2: Vehicle frame → camera frame (inverse of camera → vehicle)
    cam_translation = np.array(camera_calib['translation'])
    cam_rotation = np.array(camera_calib['rotation']) # quaternion [w, x, y, z] (NuScenes format)
    ego_to_cam = np.linalg.inv(transform_matrix(cam_translation, cam_rotation))
    
    # Combined: LiDAR frame → camera frame
    lidar_to_cam_transform = ego_to_cam @ lidar_to_ego
    
    print("\nLiDAR to Camera Transformation Matrix (4x4):")
    print("(Transforms 3D LiDAR coords → 3D camera coords)")
    print(lidar_to_cam_transform)
    
    return cam_intrinsic, lidar_to_cam_transform


def quaternion_to_rotation_matrix(q):
    """
    Convert quaternion to 3x3 rotation matrix using numpy
    
    inputs:
        q: quaternion [w, x, y, z] (NuScenes format)
    
    Returns:
        3x3 rotation matrix
    """
    w, x, y, z = q[0], q[1], q[2], q[3]
    
    # Rotation matrix from quaternion formula
    # https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
    R = np.array([
        [1 - 2*(y**2 + z**2),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x**2 + z**2),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
    ])
    
    return R

def transform_matrix(translation, rotation_quat):
    """
    Build a 4x4 homogeneous transformation matrix from translation and quaternion
    
    Args:
        translation: [x, y, z] translation vector
        rotation_quat: [w, x, y, z] quaternion (NuScenes format)
    
    Returns:
        4x4 transformation matrix
    """
    # start with 4x4 identity matrix
    mat = np.eye(4)
    # rotation part (top-left 3x3)
    mat[:3, :3] = quaternion_to_rotation_matrix(rotation_quat)
    # translation part (top-right 3x1)
    mat[:3, 3] = translation
    return mat


def project_lidar_to_camera(lidar_points, cam_intrinsic, lidar_to_cam_transform, image_shape):
    """
    Project LiDAR points onto camera image plane
    
    Args:
        lidar_points: LiDAR point cloud (4 x N array: x, y, z, intensity)
        cam_intrinsic: Camera intrinsic matrix (3x3)
        lidar_to_cam_transform: Transformation from LiDAR to camera frame (4x4)
        image_shape: (height, width) of the camera image
    
    Returns:
        points_2d: Projected 2D points (2 x M array: u, v pixel coordinates)
        depths: Depth values for each projected point (M array)
        mask: Boolean mask indicating which points are in camera view (N array)
    """
    # transform LiDAR points to camera frame
    points_3d_lidar = np.vstack((lidar_points[:3, :], np.ones(lidar_points.shape[1])))
    
    # applying transformation: LiDAR frame → camera frame
    points_3d_cam = lidar_to_cam_transform @ points_3d_lidar
    
    # filter points behind the camera (negative depth)
    # in camera frame: X=right, Y=down, Z=forward (depth)
    depths = points_3d_cam[2, :]
    mask = depths > 0  # Keep only points in front of camera
    
    # project 3D points to 2D image plane using camera intrinsics
    # refer notes for more details
    points_2d = view_points(points_3d_cam[:3, :], cam_intrinsic, normalize=True)[:2, :]
    
    # filter points outside image boundaries
    image_h, image_w = image_shape[:2]
    mask = mask & (points_2d[0, :] >= 0) & (points_2d[0, :] < image_w)
    mask = mask & (points_2d[1, :] >= 0) & (points_2d[1, :] < image_h)
    
    print(f"\nProjection Statistics:")
    print(f"  Total LiDAR points: {lidar_points.shape[1]}")
    print(f"  Points in camera view: {mask.sum()}")
    print(f"  Percentage visible: {100 * mask.sum() / lidar_points.shape[1]:.1f}%")
    
    return points_2d[:, mask], depths[mask], mask


def visualize_lidar_on_camera(camera_image, points_2d, depths, save_path=None):
    """
    Visualize LiDAR points overlaid on camera image
    
    Args:
        camera_image: Camera image as numpy array
        points_2d: Projected 2D points (2 x N array)
        depths: Depth values for coloring (N array)
        save_path: Optional path to save the visualization
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    # Left: Original camera image
    ax1.imshow(camera_image)
    ax1.set_title('Original CAM_FRONT Image', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Right: Camera image with LiDAR overlay
    ax2.imshow(camera_image)
    
    # Overlay LiDAR points colored by depth
    scatter = ax2.scatter(
        points_2d[0, :],  # x (horizontal pixel coordinate)
        points_2d[1, :],  # y (vertical pixel coordinate)
        c=depths,         # color by depth
        cmap='jet',       # colormap (blue=close, red=far)
        s=2,              # point size
        alpha=0.5         # transparency
    )
    
    ax2.set_title('LiDAR Points Projected on Camera Image (colored by depth)', 
                  fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('Depth (meters)', fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {save_path}")
    
    plt.show()


def visualize_depth_map(camera_image, points_2d, depths, save_path=None):
    """
    Create a detailed depth map visualization
    
    Args:
        camera_image: Camera image as numpy array
        points_2d: Projected 2D points (2 x N array)
        depths: Depth values (N array)
        save_path: Optional path to save the visualization
    """
    fig = plt.figure(figsize=(20, 6))
    
    # Plot 1: Camera image only
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(camera_image)
    ax1.set_title('Camera Image', fontsize=13, fontweight='bold')
    ax1.axis('off')
    
    # Plot 2: LiDAR overlay with depth coloring
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(camera_image, alpha=0.6)
    scatter = ax2.scatter(
        points_2d[0, :], points_2d[1, :],
        c=depths, cmap='jet', s=3, alpha=0.8, vmin=0, vmax=50
    )
    ax2.set_title('LiDAR Projection (Depth Colored)', fontsize=13, fontweight='bold')
    ax2.axis('off')
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Depth (m)', fontsize=10)
    
    # Plot 3: Depth histogram
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.hist(depths, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax3.set_xlabel('Depth (meters)', fontsize=11)
    ax3.set_ylabel('Number of Points', fontsize=11)
    ax3.set_title('Depth Distribution', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.axvline(depths.mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {depths.mean():.1f}m')
    ax3.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Depth visualization saved to: {save_path}")
    
    plt.show()


def main():
    # initialize NuScenes
    dataroot = 'data/nuscenes'
    version = 'v1.0-mini'
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
    
    # loading LiDAR and camera data
    sample_idx = 0
    lidar_points, camera_image, sample, lidar_token, camera_token = load_lidar_camera_sample(
        nusc, sample_idx=sample_idx
    )
    
    # getting calibration matrices
    print("GET CALIBRATION MATRICES:")
    cam_intrinsic, lidar_to_cam_transform = get_calibration_matrices(
        nusc, lidar_token, camera_token
    )
    
    # project LiDAR to camera
    print("PROJECT 3D LIDAR POINTS TO 2D IMAGE:")
    points_2d, depths, mask = project_lidar_to_camera(
        lidar_points, 
        cam_intrinsic, 
        lidar_to_cam_transform,
        camera_image.shape
    )
    
    print("\nGenerating overlay visualization...")
    visualize_lidar_on_camera(
        camera_image,
        points_2d,
        depths,
        save_path='outputs/output_projection_overlay.png'
    )
    
    print("\nGenerating depth map visualization...")
    visualize_depth_map(
        camera_image,
        points_2d,
        depths,
        save_path='outputs/output_projection_depth.png'
    )
    
    print("SUCCESS! LiDAR points projected onto camera image.")

    return {
        'nusc': nusc,
        'sample': sample,
        'lidar_points': lidar_points,
        'camera_image': camera_image,
        'points_2d': points_2d,
        'depths': depths,
        'cam_intrinsic': cam_intrinsic,
        'lidar_to_cam_transform': lidar_to_cam_transform
    }


if __name__ == "__main__":
    data = main()

