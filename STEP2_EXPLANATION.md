# Step 2: Project LiDAR onto Camera - Explained

## What This Script Does

The `project_lidar_to_camera.py` script demonstrates **sensor fusion** by overlaying 3D LiDAR points onto a 2D camera image. This is a crucial step in autonomous vehicle perception.

---

## The Math Behind It

### **The Problem:**
- LiDAR gives us 3D points in **LiDAR coordinate frame** (x, y, z)
- Camera gives us a 2D image in **pixel coordinates** (u, v)
- We need to transform LiDAR points → Camera frame → Image pixels

### **The Solution: 3-Step Transformation**

```
3D LiDAR coords → 3D Camera coords → 2D Image pixels
   (4x4 matrix)      (3x3 matrix)
```

---

## Key Functions Explained

### 1. `get_calibration_matrices()`

**Purpose:** Get the transformation matrices from the dataset

**Outputs:**
- **Camera Intrinsic Matrix (3×3)**: Projects 3D camera points to 2D pixels
  ```
  [fx  0  cx]     fx, fy = focal lengths (zoom)
  [0  fy  cy]     cx, cy = image center (principal point)
  [0   0   1]
  ```

- **LiDAR-to-Camera Transform (4×4)**: Converts LiDAR 3D → Camera 3D
  ```
  [R R R tx]      R = rotation matrix (3×3)
  [R R R ty]      t = translation vector (3×1)
  [R R R tz]
  [0 0 0  1]
  ```

**Why two transforms?**
- First: Rotate & translate from LiDAR position to camera position
- Second: Project 3D camera points to 2D image

---

### 2. `transform_matrix()`

**Purpose:** Build a 4×4 homogeneous transformation matrix

**Inputs:**
- Translation: `[x, y, z]` - how far to move
- Rotation: `[w, x, y, z]` - quaternion (how much to rotate)

**Output:** Combined 4×4 matrix

**Why quaternions?** 
They're a compact way to represent 3D rotations without gimbal lock.

---

### 3. `project_lidar_to_camera()`

**Purpose:** The main projection algorithm

**Step-by-step:**

1. **Add homogeneous coordinate**
   ```python
   [x, y, z] → [x, y, z, 1]
   ```
   This allows us to use matrix multiplication for translation + rotation

2. **Transform to camera frame**
   ```python
   points_3d_cam = lidar_to_cam_transform @ points_3d_lidar
   ```

3. **Filter points behind camera**
   ```python
   depths = points_3d_cam[2, :]  # Z coordinate in camera frame
   mask = depths > 0              # Keep only positive depth
   ```

4. **Project to 2D pixels**
   ```python
   points_2d = cam_intrinsic @ points_3d_cam
   points_2d = points_2d / points_2d[2]  # Normalize by depth
   ```
   This is the **pinhole camera model**

5. **Filter points outside image**
   ```python
   Keep only: 0 ≤ u < width AND 0 ≤ v < height
   ```

**Output:**
- 2D pixel coordinates `(u, v)` for each visible LiDAR point
- Depth values for coloring

---

### 4. `visualize_lidar_on_camera()`

**Purpose:** Overlay LiDAR points on the camera image

**Visualization:**
- Each LiDAR point becomes a colored dot on the image
- Color = depth (distance from camera)
  - **Blue** = close objects
  - **Red** = far objects

**What to look for:**
- ✅ Points should align with objects (cars, buildings, roads)
- ✅ Cars should be covered in blue/green dots
- ✅ Distant buildings should be red/orange dots
- ❌ If misaligned, calibration might be wrong

---

### 5. `visualize_depth_map()`

**Purpose:** Create a 3-panel visualization

**Panels:**
1. **Original camera image** - what the camera sees
2. **LiDAR overlay** - colored dots on the image
3. **Depth histogram** - distribution of distances

**Insights from histogram:**
- Peak at 0-10m: nearby vehicles and road
- Tail to 50m+: distant objects
- Shows LiDAR's effective range

---

## Coordinate Systems

### LiDAR Frame (sensor on car roof):
- **X**: Forward (front of car)
- **Y**: Left (driver side)
- **Z**: Up (sky)
- **Origin**: LiDAR sensor position

### Camera Frame (camera on windshield):
- **X**: Right (passenger side) ⚠️ NOTE: different from LiDAR!
- **Y**: Down (toward ground) ⚠️ NOTE: flipped from LiDAR!
- **Z**: Forward (depth into scene)
- **Origin**: Camera optical center

### Image Frame (pixels):
- **U (x)**: Horizontal pixel (0 = left, max = right)
- **V (y)**: Vertical pixel (0 = top, max = bottom)
- **Origin**: Top-left corner

---

## Expected Output

Running the script produces:

1. **`output_projection_overlay.png`**
   - Side-by-side comparison
   - Left: Original camera image
   - Right: LiDAR dots overlaid on image

2. **`output_projection_depth.png`**
   - 3-panel visualization
   - Shows camera, overlay, and depth distribution

3. **Console output:**
   - Calibration matrices
   - Point counts and statistics
   - Projection success metrics

---

## How to Run

```bash
cd /Users/nehask/Desktop/AV/bevfusion
python project_lidar_to_camera.py
```

**Requirements:**
Make sure you've installed:
```bash
pip install nuscenes-devkit numpy matplotlib pillow pyquaternion
```

---

## What's Next? (Step 3)

After verifying alignment, you can:
1. Use annotations to get vehicle bounding boxes
2. Cluster LiDAR points inside bounding boxes
3. Fuse 2D detections with 3D LiDAR data
4. Create a Bird's Eye View (BEV) map

---

## Troubleshooting

**Problem:** Points don't align with objects
- **Fix:** Check if calibration files are correct in NuScenes data

**Problem:** No points visible
- **Fix:** Check if mask is too restrictive (increase image bounds)

**Problem:** "pyquaternion not found"
- **Fix:** `pip install pyquaternion`

**Problem:** Points aligned but colors look wrong
- **Fix:** Adjust colormap limits in `vmin`/`vmax` parameters

