import time
import cv2 as cv
import cv2.aruco as aruco

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

cap = cv.VideoCapture(0)
time.sleep(2)

nframes = 25
snapshot = None
final_ids = None
final_corners = None

for _ in range(nframes):
    ret, frame = cap.read()
    if not ret:
        continue
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    corners, ids, _rejected = detector.detectMarkers(gray)
    
    # Always keep the latest frame as a fallback
    snapshot = frame 
    
    # If we successfully detect a marker, save the data and break the loop
    if ids is not None:
        final_ids = ids
        final_corners = corners
        break 

cap.release()

print(f"Detected IDs: {final_ids.flatten().tolist() if final_ids is not None else []}")

if final_ids is not None:
    aruco.drawDetectedMarkers(snapshot, final_corners, final_ids)

cv.imwrite("detected_cubes_snapshot.jpg", snapshot)
print("Saved detected_cubes_snapshot.jpg")S