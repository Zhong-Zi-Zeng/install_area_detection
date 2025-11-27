# Install Area Detection

A ROS 2 package for detecting and recognizing fans installation areas.

## Description

This package provides tools and algorithms to detect and recognize suitable areas for fan installation.

## Prerequisites

- Ubuntu 22.04 or later
- ROS 2 Humble
- Python 3.8+

## Installation

### 1. Clone the repository

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone https://github.com/Zhong-Zi-Zeng/install_area_detection.git
```

### 2. Install dependencies

```bash
cd ~/ros2_ws
rosdep update
rosdep install --from-paths src --ignore-src -r -y
```

### 3. Build the package

```bash
cd ~/ros2_ws
colcon build --packages-select install_area_detection
```

### 4. Source the setup file

```bash
source ~/ros2_ws/install/setup.bash
```

## Usage

### Launch with parameters

```bash
ros2 launch install_area_detection detector.launch.py
```

## Configuration

You can customize the detection parameters by modifying the configuration file located at `config/params.yaml`.

## Topics

### Published Topics

 
### Subscribed Topics

- `/depth`: Depth image topic
- `/image_rect`: RGB image topic


## Contact

Zhong-Zi-Heng - ximen0806449@gmail.com