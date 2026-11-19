import cv2
from cv2_enumerate_cameras import enumerate_cameras

camera_info = {}

for cam in enumerate_cameras(cv2.CAP_MSMF):
    camera_info[cam.index] = cam.name
    print(f"Index: {cam.index} | Name: {cam.name}")

# Open the default camera
cam = cv2.VideoCapture(0)

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(frame_width, frame_height)

while True:
    ret, frame = cam.read()

    # Display the captured frame
    cv2.imshow('Camera', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and writer objects
cam.release()
# out.release()
cv2.destroyAllWindows()