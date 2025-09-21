import cv2
import time

def capture_images(interval=60, camera_index=0):
    # Open webcam
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    img_count = 0
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Save image
            filename = f"capture_{img_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Image saved: {filename}")

            img_count += 1

            # Wait for interval
            time.sleep(interval)

    except KeyboardInterrupt:
        print("Capture stopped by user.")

    # Release camera
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_images(interval=60)  # 60 seconds interval
