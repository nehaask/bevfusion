import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
import os

def load_lidar_camera_sample(nusc, sample_idx=0):
    
    # get the sample
    sample = nusc.sample[sample_idx]
    
    print(f"Sample {sample_idx}: {sample['token']}")
    print(f"Scene: {nusc.get('scene', sample['scene_token'])['name']}")
    print(f"Timestamp: {sample['timestamp']}")
    
    # get LiDAR data token
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_sample = nusc.get('sample_data', lidar_token)
    
    # load LiDAR point cloud
    lidar_path = os.path.join(nusc.dataroot, lidar_sample['filename'])
    lidar_pointcloud = LidarPointCloud.from_file(lidar_path)
    lidar_data = lidar_pointcloud.points  # 4 x N array (x, y, z, intensity)

    # get camera data token
    camera_token = sample['data']['CAM_FRONT']
    camera_sample = nusc.get('sample_data', camera_token)
    
    # load camera image
    camera_path = os.path.join(nusc.dataroot, camera_sample['filename'])
    camera_image = np.array(Image.open(camera_path)) # RGB (H × W × 3)
    
    print(f"LiDAR Data:")
    print(f"  - Shape: {lidar_data.shape}")
    print(f"  - Number of points: {lidar_data.shape[1]}")
    print(f"  - X range: [{lidar_data[0].min():.2f}, {lidar_data[0].max():.2f}]")
    print(f"  - Y range: [{lidar_data[1].min():.2f}, {lidar_data[1].max():.2f}]")
    print(f"  - Z range: [{lidar_data[2].min():.2f}, {lidar_data[2].max():.2f}]")
    print(f"  - File: {lidar_sample['filename']}")
    
    print(f"\nCamera Data:")
    print(f"  - Shape: {camera_image.shape}")
    print(f"  - Resolution: {camera_image.shape[1]} x {camera_image.shape[0]}")
    print(f"  - File: {camera_sample['filename']}")
    
    return lidar_data, camera_image, sample, lidar_token, camera_token


def visualize_lidar_camera(lidar_data, camera_image, save_path=None):
    fig = plt.figure(figsize=(16, 6))
    
    # plot camera image
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(camera_image)
    ax1.set_title('CAM_FRONT Image', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # plot LiDAR BEV (top-down)
    ax2 = fig.add_subplot(1, 2, 2)

    # color by height (z value)
    x, y, z = lidar_data[0:3]
    scatter = ax2.scatter(x, y, c=z, cmap='viridis', s=1, alpha=0.5)
    ax2.set_xlabel('X (meters)', fontsize=12)
    ax2.set_ylabel('Y (meters)', fontsize=12)
    ax2.set_title('LIDAR_TOP Bird\'s Eye View (colored by height)', fontsize=14, fontweight='bold')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    # adding colorbar
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Height Z (meters)', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {save_path}")
    
    plt.show()


def main():
    """Main function to load and visualize NuScenes data"""
    
    # initialize NuScenes dataset
    dataroot = 'data/nuscenes'
    version = 'v1.0-mini'
    
    print(f"Initializing NuScenes {version} dataset...")
    print(f"Data root: {dataroot}\n")
    
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
    
    print(f"\nDataset loaded successfully!")
    print(f"Number of scenes: {len(nusc.scene)}")
    print(f"Number of samples: {len(nusc.sample)}")
    print(f"Number of sample_data: {len(nusc.sample_data)}")
    
    # Load first sample (index 0)
    sample_idx = 0
    lidar_data, camera_image, sample, lidar_token, camera_token = load_lidar_camera_sample(
        nusc, sample_idx=sample_idx
    )
    
    # visualization
    print("\nGenerating visualization...")
    visualize_lidar_camera(
        lidar_data, 
        camera_image, 
        save_path=f'outputs/output_sample_{sample_idx}.png'
    )

if __name__ == "__main__":
    main()

