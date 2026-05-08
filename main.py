import time
import traceback
import os
import sys
import numpy as np
import cv2 as cv
import cv2.aruco as aruco

# Ensure paths for robot and kinematics modules are accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../funrobo_hiwonder")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "./")))

from funrobo_hiwonder.core.hiwonder import HiwonderRobot
from funrobo_kinematics.core.fiveDOFrrmc import FiveDOFRobot
import funrobo_kinematics.core.utils as ut

# --- 1. Global Setup & Calibration ---
with np.load('calibration_data.npz') as data:
    mtx = data['mtx']
    dist = data['dist']

MARKER_SIZE = 0.0254 # 1 inch
MY_CUBE_IDS = [1, 2, 3]
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

# 3D model points for solvePnP
obj_points = np.array([
    [-MARKER_SIZE/2,  MARKER_SIZE/2, 0],
    [ MARKER_SIZE/2,  MARKER_SIZE/2, 0],
    [ MARKER_SIZE/2, -MARKER_SIZE/2, 0],
    [-MARKER_SIZE/2, -MARKER_SIZE/2, 0]
], dtype=np.float32)

def detect_waypoints(cap, model, curr_joints_rad, exclude_ids=None, nframes=25, z_table=0.018):
    """Detect tags and return {id: [x,y,z]} in base frame (median over frames)."""
    if exclude_ids is None:
        exclude_ids = set()

    # Compute H_base_to_cam for current arm pose.
    H_cumulative, _ = model.compute_transformation_matrices(curr_joints_rad[:5])
    H_base_to_wrist = H_cumulative[4]

    R_cam = np.array([
        [0, 0, 1],
        [1, 0, 0],
        [0, -1, 0]
    ], dtype=float)
    H_cam_offset = np.eye(4)
    H_cam_offset[:3, :3] = R_cam
    H_cam_offset[:3, 3] = [-0.04, 0, 0]

    H_base_to_cam = H_base_to_wrist @ H_cam_offset
    print("Camera world position:", H_base_to_cam[:3, 3])
    print("Camera Z-axis in base frame (optical axis):", H_base_to_cam[:3, 2])

    samples_per_id = {mid: [] for mid in MY_CUBE_IDS if mid not in exclude_ids}
    for _ in range(nframes):
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        corners, ids, _rejected = detector.detectMarkers(gray)
        if ids is None:
            continue

        for i in range(len(ids)):
            marker_id = int(ids[i][0])
            if marker_id not in samples_per_id:
                continue

            ok, rvec, tvec = cv.solvePnP(
                obj_points, corners[i], mtx, dist, flags=cv.SOLVEPNP_IPPE_SQUARE
            )
            if not ok:
                continue

            H_cam_to_cube = np.eye(4)
            H_cam_to_cube[:3, :3] = cv.Rodrigues(rvec)[0]
            H_cam_to_cube[:3, 3] = [tvec[0][0], tvec[1][0], tvec[2][0]]

            H_base_to_cube = H_base_to_cam @ H_cam_to_cube
            samples_per_id[marker_id].append(H_base_to_cube[:3, 3].copy())

    waypoints = {}
    for marker_id, samples in samples_per_id.items():
        if len(samples) == 0:
            continue
        arr = np.vstack(samples)
        med = np.median(arr, axis=0)
        waypoints[marker_id] = [float(med[0]), float(med[1]), float(z_table)]
        print(f"ID {marker_id}: used {len(samples)}/{nframes} samples")
        print(f"ID {marker_id} median base frame: X:{med[0]:.3f}, Y:{med[1]:.3f}, Z:{med[2]:.3f}")
        print(f"ID {marker_id} (used) base frame: X:{med[0]:.3f}, Y:{med[1]:.3f}, Z:{z_table:.3f}")

    return waypoints

def main():
    robot = None
    try:
        # Initialize Robot and Kinematics Model
        robot = HiwonderRobot() #
        model = FiveDOFRobot() #
       
        dt = 2
        block_count = 0

        # Get current state for the "Scanning Pose"
        curr_joint_values = robot.get_joint_values() #
        curr_joints_rad = [np.deg2rad(theta) for theta in curr_joint_values] #
       
        # Dictionary to store calculated Base-Frame waypoints for each ID
        HI_waypoints = {}

        # --- 2. Image Capture ---
        cap = cv.VideoCapture(0)
        time.sleep(2) # Camera warm-up for exposure
       
        # --- 3. Marker Detection & Frame Transformation ---
        # Compute H_base_to_cam ONCE (transform instability should not come from here)
        H_cumulative, _ = model.compute_transformation_matrices(curr_joints_rad[:5])
        H_base_to_wrist = H_cumulative[4]  # wrist / joint-4 frame (camera bracket)

        # Wrist -> Camera mount rotation (keep this as your calibrated/guessed fixed matrix).
        R_cam = np.array([
            [0, 0, 1],
            [1, 0, 0],
            [0, -1, 0]
        ], dtype=float)
        H_cam_offset = np.eye(4)
        H_cam_offset[:3, :3] = R_cam
        H_cam_offset[:3, 3] = [-0.04, 0, 0]  # meters, lens offset from wrist pivot (in wrist frame)

        H_base_to_cam = H_base_to_wrist @ H_cam_offset
        print("Camera world position:", H_base_to_cam[:3, 3])
        print("Camera Z-axis in base frame (optical axis):", H_base_to_cam[:3, 2])

        # Robust detection: take multiple frames and median the result per marker id.
        samples_per_id = {mid: [] for mid in MY_CUBE_IDS}
        nframes = 25
        for _ in range(nframes):
            ret, frame = cap.read()
            if not ret:
                continue
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            corners, ids, _rejected = detector.detectMarkers(gray)
            if ids is None:
                continue

            for i in range(len(ids)):
                marker_id = int(ids[i][0])
                if marker_id not in MY_CUBE_IDS:
                    continue

                # More stable for planar square tags than the default iterative PnP.
                ok, rvec, tvec = cv.solvePnP(
                    obj_points, corners[i], mtx, dist, flags=cv.SOLVEPNP_IPPE_SQUARE
                )
                if not ok:
                    continue

                H_cam_to_cube = np.eye(4)
                H_cam_to_cube[:3, :3] = cv.Rodrigues(rvec)[0]
                H_cam_to_cube[:3, 3] = [tvec[0][0], tvec[1][0], tvec[2][0]]

                H_base_to_cube = H_base_to_cam @ H_cam_to_cube
                samples_per_id[marker_id].append(H_base_to_cube[:3, 3].copy())

        cap.release()

        for marker_id, samples in samples_per_id.items():
            if len(samples) == 0:
                continue

            arr = np.vstack(samples)  # (N,3)
            med = np.median(arr, axis=0)
            # Temporary: if the cube sits on the table and base Z=0 is the t
            able plane,
            # use vision for X/Y and clamp Z to the plane.
            Z_TABLE = 0.018
            HI_waypoints[marker_id] = [float(med[0]), float(med[1]), float(med[2] - 0.29)]

            print(f"ID {marker_id}: used {len(samples)}/{nframes} samples")
            print(f"ID {marker_id} median base frame: X:{med[0]:.3f}, Y:{med[1]:.3f}, Z:{med[2]:.3f}")
        if len(HI_waypoints) == 0:
            print("No valid markers detected across frames.")
            return

        # --- 4. PICK then DROP ---
        def set_gripper_deg(angle_deg: float, duration_s: float = 0.6):
            """Command gripper by setting joint 6 (index 5)."""
            joints = list(curr_joints_rad)
            joints[5] = float(np.deg2rad(angle_deg))
            robot.set_joint_values(joints, duration=duration_s, radians=True)
            return joints

        open_deg = float(getattr(robot, "open_gripper_angle", -110))
        close_deg = float(getattr(robot, "close_gripper_angle", 0))
        avoid_ids = []
        for id in sorted(HI_waypoints.keys()):
            pick_id = id
            avoid_ids.append(pick_id)
            pick_pos = HI_waypoints[pick_id]
            print(f"Picking ID {pick_id}...")

            # Open gripper, approach, close, lift
            curr_joints_rad = set_gripper_deg(open_deg, duration_s=0.6)
            time.sleep(0.6)

            pick_path = [
                [pick_pos[0], pick_pos[1], pick_pos[2] + 0.005],
                [pick_pos[0], pick_pos[1], pick_pos[2]],
                [pick_pos[0], pick_pos[1], pick_pos[2] + 0.005],
            ]
            for idx, point in enumerate(pick_path):
                ee = ut.EndEffector()
                ee.x, ee.y, ee.z = point
                joint_values = model.calc_numerical_ik(ee, curr_joints_rad[:5])
                final_joints = list(joint_values) + [curr_joints_rad[5]]
                robot.set_joint_values(final_joints, duration=1.5, radians=True)
                time.sleep(dt)
                curr_joints_rad = list(final_joints)
                if idx == 1:
                    curr_joints_rad = set_gripper_deg(close_deg, duration_s=0.6)
                    time.sleep(0.6)

            # Go HOME before rotating joint 1 (safe clearance)
            robot.move_to_home_position()
            time.sleep(2.0)


            curr_joint_values = robot.get_joint_values()
            curr_joints_rad = [np.deg2rad(theta) for theta in curr_joint_values]

            # Rotate joint 1 counter-clockwise by +90 degrees
            rotated = list(curr_joints_rad)
            rotated[0] = float(rotated[0] + (np.pi / 2))
            robot.set_joint_values(rotated, duration=1.5, radians=True)
            time.sleep(2.0)
            curr_joints_rad = rotated

            # Detect a different tag for DROP
            cap2 = cv.VideoCapture(0)
            time.sleep(1.0)
            drop_waypoints = detect_waypoints(cap2, model, curr_joints_rad, exclude_ids={pick_id}, nframes=25, z_table=0.01)
            cap2.release()

            if len(drop_waypoints) == 0:
                print("No drop tag found. Returning home.")
                robot.move_to_home_position()
                time.sleep(2.0)
                return

            drop_id = sorted(drop_waypoints.keys())[0]
            drop_pos = drop_waypoints[drop_id]
            print(f"Dropping at ID {drop_id}...")

            drop_path = [
                [drop_pos[0], drop_pos[1], drop_pos[2] + block_count * .04],
                [drop_pos[0], drop_pos[1], drop_pos[2] + block_count * .04],
            ]
            for point in drop_path:
                ee = ut.EndEffector()
                ee.x, ee.y, ee.z = point
                joint_values = model.calc_numerical_ik(ee, curr_joints_rad[:5])
                final_joints = list(joint_values) + [curr_joints_rad[5]]
                robot.set_joint_values(final_joints, duration=1.5, radians=True)
                time.sleep(dt)
                curr_joints_rad = list(final_joints)

            # Release and go HOME
            curr_joints_rad = set_gripper_deg(open_deg, duration_s=0.6)
            time.sleep(0.6)
            ee = ut.EndEffector()
            ee.x, ee.y, ee.z = [drop_pos[0], drop_pos[1], drop_pos[2] + 0.135]
            joint_values = model.calc_numerical_ik(ee, curr_joints_rad[:5])
            final_joints = list(joint_values) + [curr_joints_rad[5]]
            robot.set_joint_values(final_joints, duration=1.5, radians=True)
            time.sleep(dt)
            curr_joints_rad = list(final_joints)
            time.sleep(0.6)
            robot.move_to_home_position()
            time.sleep(2.0)
            curr_joint_values = robot.get_joint_values()  # ADD THIS
            curr_joints_rad = [np.deg2rad(theta) for theta in curr_joint_values]  # AND THIS
            block_count = block_count + 1

        
               
    except Exception as e:
        print(f"[ERROR] {e}")
        traceback.print_exc()
    finally:
        if robot:
            robot.shutdown_robot() #

if __name__ == "__main__":
    main()