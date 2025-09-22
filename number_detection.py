import cv2
import time
import re
import os
import easyocr

def extract_digits(text):
    """Return the first group of digits found (or None)."""
    if not text:
        return None
    groups = re.findall(r'\d+', text)
    return groups[0] if groups else None

def ocr_with_easyocr(frame, reader):
    # EasyOCR expects images in numpy array (BGR->RGB)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    try:
        results = reader.readtext(img_rgb, detail=0)  # detail=0 -> only text strings
    except Exception as e:
        print("EasyOCR error:", e)
        return None, None
    joined = " ".join(results).strip()
    digits = extract_digits(joined)
    return joined, digits

def capture_and_detect_number(interval=60, camera_index=0, save_dir="captures"):
    os.makedirs(save_dir, exist_ok=True)
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: cannot open camera")
        return

    # Initialize EasyOCR reader once
    reader = easyocr.Reader(['en'], gpu=False)

    img_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            filename = os.path.join(save_dir, f"capture_{img_count}.jpg")
            cv2.imwrite(filename, frame)
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Saved {filename}")

            # OCR with EasyOCR
            raw_text, digits = ocr_with_easyocr(frame, reader=reader)
            if digits:
                print("EasyOCR detected digits:", digits, "| raw:", raw_text)
            else:
                print("No digits found by EasyOCR.")

            img_count += 1
            time.sleep(interval)

    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example: run with 10-second interval for testing
    capture_and_detect_number(interval=10, camera_index=0)