import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
import os


def load_lidar_camera_sample(nusc, sample_idx=0):
    """
    Load synchronized LiDAR and camera data for a given sample
    
    Args:
        nusc: NuScenes instance
        sample_idx: Index of the sample to load (0 to len(nusc.sample)-1)
    
    Returns:
        lidar_data: LiDAR point cloud (4 x N array: x, y, z, intensity)
        camera_image: Camera image as numpy array
        sample_record: Sample metadata
        lidar_token: LiDAR sample_data token
        camera_token: Camera sample_data token
    """
    # Get the sample
    sample = nusc.sample[sample_idx]
    
    print(f"\n{'='*60}")
    print(f"Sample {sample_idx}: {sample['token']}")
    print(f"Scene: {nusc.get('scene', sample['scene_token'])['name']}")
    print(f"Timestamp: {sample['timestamp']}")
    print(f"{'='*60}\n")
    
    # Get LiDAR data token
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_sample = nusc.get('sample_data', lidar_token)
    
    # Get Camera data token
    camera_token = sample['data']['CAM_FRONT']
    camera_sample = nusc.get('sample_data', camera_token)
    
    # Load LiDAR point cloud
    lidar_path = os.path.join(nusc.dataroot, lidar_sample['filename'])
    lidar_pointcloud = LidarPointCloud.from_file(lidar_path)
    lidar_data = lidar_pointcloud.points  # 4 x N array (x, y, z, intensity)
    
    # Load camera image
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
    """
    Visualize LiDAR point cloud and camera image side by side
    
    Args:
        lidar_data: LiDAR point cloud (4 x N array)
        camera_image: Camera image as numpy array
        save_path: Optional path to save the visualization
    """
    fig = plt.figure(figsize=(16, 6))
    
    # Plot camera image
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(camera_image)
    ax1.set_title('CAM_FRONT Image', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Plot LiDAR bird's eye view (top-down)
    ax2 = fig.add_subplot(1, 2, 2)
    
    # Get x, y coordinates
    x = lidar_data[0]
    y = lidar_data[1]
    z = lidar_data[2]
    
    # Color by height (z value)
    scatter = ax2.scatter(x, y, c=z, cmap='viridis', s=1, alpha=0.5)
    ax2.set_xlabel('X (meters)', fontsize=12)
    ax2.set_ylabel('Y (meters)', fontsize=12)
    ax2.set_title('LIDAR_TOP Bird\'s Eye View (colored by height)', fontsize=14, fontweight='bold')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Height Z (meters)', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {save_path}")
    
    plt.show()


def main():
    """Main function to load and visualize NuScenes data"""
    
    # Initialize NuScenes
    dataroot = '/Users/nehask/Desktop/AV/bevfusion/data/nuscenes'
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
    
    # Visualize
    print("\nGenerating visualization...")
    visualize_lidar_camera(
        lidar_data, 
        camera_image, 
        save_path='output_sample_0.png'
    )
    
    print("\n" + "="*60)
    print("SUCCESS! Data loaded and visualized.")
    print("="*60)


if __name__ == "__main__":
    main()

