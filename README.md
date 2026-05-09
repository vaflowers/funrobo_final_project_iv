# Automated Cube Detection and Retrieval Using a 5-DOF Hiwonder Robot

This repository contains the implementation of an autonomous manipulation system for the 5-DOF Hiwonder robot. The goal of this project is to detect an AprilTag on an object, transform the 2D pixel coordinates into 3D workspace coordinates, and execute a precise retrieval sequence using inverse kinematics.

<p align="center">
  <img src="images/demo_video.gif" width="600" />
</p>


## Implementation Instructions

Follow these steps to set up and execute the system:

1.  **Acquire Robot System Files:** Ensure all necessary Hiwonder robot driver files and the `funrobo_kinematics` module are present in the workspace.
2.  **Install Requirements:**
     ```bash
    pip install -r requirements.txt
    ```
3.  **Activate Environment:**
     ```bash
    conda activate funrobo_hw
    ```
4.  **Calibrate Camera:** Run the calibration script to generate the intrinsic matrix.
    ```bash
    python3 camera_calibration.py
    ```
5.  **Run System:** Execute the main control loop.
    ```bash
    python3 main.py
    ```

## Project Documentation 

Follow the link to view the technical report.

https://docs.google.com/document/d/1It2Th18x3OTXK4cxnojgHj7ktT9yjoGg_4Uf_scdwWw/edit?usp=sharing
