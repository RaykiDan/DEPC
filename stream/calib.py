import cv2
import numpy as np
import math
import os


class WebcamCalibration:
    def __init__(self):
        # Checkerboard dimensions (internal corners)
        # Default: 9x6 for standard printable checkerboard
        self.CHECKERBOARD = (8, 6)

        # Termination criteria for corner refinement
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Prepare object points (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
        self.objp = np.zeros((self.CHECKERBOARD[0] * self.CHECKERBOARD[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.CHECKERBOARD[0], 0:self.CHECKERBOARD[1]].T.reshape(-1, 2)

        # Arrays to store object points and image points
        self.objpoints = []  # 3D points in real world space
        self.imgpoints = []  # 2D points in image plane

        self.camera_matrix = None
        self.dist_coeffs = None
        self.img_shape = None

    def capture_calibration_images(self, camera_id=0, num_images=20):
        """
        Capture calibration images from webcam
        """
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            print("[ERROR] Cannot open webcam")
            return False

        print("=" * 60)
        print("WEBCAM FOV CALIBRATION TOOL")
        print("=" * 60)
        print(f"Checkerboard pattern: {self.CHECKERBOARD[0]}x{self.CHECKERBOARD[1]} internal corners")
        print(f"Target images: {num_images}")
        print("\nInstructions:")
        print("1. Print a checkerboard pattern (you can find one online)")
        print("2. Hold the checkerboard in front of the camera")
        print("3. Press SPACE when corners are detected (green lines)")
        print("4. Move the checkerboard to different positions/angles")
        print("5. Press ESC to finish and calculate FOV")
        print("=" * 60)

        captured = 0

        while captured < num_images:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read frame")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.img_shape = gray.shape[::-1]

            # Find checkerboard corners
            ret_corners, corners = cv2.findChessboardCorners(
                gray,
                self.CHECKERBOARD,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
            )

            display = frame.copy()

            if ret_corners:
                # Refine corner positions
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)

                # Draw the corners
                cv2.drawChessboardCorners(display, self.CHECKERBOARD, corners_refined, ret_corners)

                # Show status
                cv2.putText(display, "Pattern detected! Press SPACE to capture",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display, "No pattern detected",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.putText(display, f"Captured: {captured}/{num_images}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow('Webcam Calibration', display)

            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                print(f"\n[INFO] Calibration stopped by user. Captured {captured} images.")
                break
            elif key == 32 and ret_corners:  # SPACE
                self.objpoints.append(self.objp)
                self.imgpoints.append(corners_refined)
                captured += 1
                print(f"[INFO] Image {captured} captured!")

        cap.release()
        cv2.destroyAllWindows()

        if captured < 10:
            print("[WARN] Need at least 10 images for good calibration!")
            return False

        return True

    def calibrate(self):
        """
        Perform camera calibration
        """
        if len(self.objpoints) == 0:
            print("[ERROR] No calibration images captured")
            return False

        print("\n[INFO] Calibrating camera...")

        ret, self.camera_matrix, self.dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints,
            self.imgpoints,
            self.img_shape,
            None,
            None
        )

        if not ret:
            print("[ERROR] Calibration failed")
            return False

        print("[INFO] Calibration successful!")
        return True

    def calculate_fov(self):
        """
        Calculate FOV from camera matrix
        """
        if self.camera_matrix is None:
            print("[ERROR] Camera not calibrated")
            return None, None

        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        width, height = self.img_shape

        # Calculate FOV in degrees
        fov_h = 2 * math.atan(width / (2 * fx)) * (180 / math.pi)
        fov_v = 2 * math.atan(height / (2 * fy)) * (180 / math.pi)

        return fov_h, fov_v

    def print_results(self):
        """
        Print calibration results
        """
        if self.camera_matrix is None:
            return

        fov_h, fov_v = self.calculate_fov()

        print("\n" + "=" * 60)
        print("CALIBRATION RESULTS")
        print("=" * 60)
        print(f"Resolution: {self.img_shape[0]}x{self.img_shape[1]}")
        print(f"\nCamera Matrix:")
        print(self.camera_matrix)
        print(f"\nDistortion Coefficients:")
        print(self.dist_coeffs)
        print(f"\nFocal Length:")
        print(f"  fx = {self.camera_matrix[0, 0]:.2f} pixels")
        print(f"  fy = {self.camera_matrix[1, 1]:.2f} pixels")
        print(f"\nPrincipal Point:")
        print(f"  cx = {self.camera_matrix[0, 2]:.2f}")
        print(f"  cy = {self.camera_matrix[1, 2]:.2f}")
        print(f"\n{'FIELD OF VIEW (FOV)':^60}")
        print("=" * 60)
        print(f"  Horizontal FOV: {fov_h:.2f}°")
        print(f"  Vertical FOV:   {fov_v:.2f}°")
        print("=" * 60)

        # Save to file
        self.save_calibration()

    def save_calibration(self, filename="webcam_calibration.npz"):
        """
        Save calibration data to file
        """
        if self.camera_matrix is None:
            return

        np.savez(
            filename,
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
            img_shape=self.img_shape
        )
        print(f"\n[INFO] Calibration data saved to '{filename}'")

    def load_calibration(self, filename="webcam_calibration.npz"):
        """
        Load calibration data from file
        """
        if not os.path.exists(filename):
            print(f"[ERROR] File '{filename}' not found")
            return False

        data = np.load(filename)
        self.camera_matrix = data['camera_matrix']
        self.dist_coeffs = data['dist_coeffs']
        self.img_shape = tuple(data['img_shape'])

        print(f"[INFO] Calibration data loaded from '{filename}'")
        return True


def main():
    calibrator = WebcamCalibration()

    print("\nWebcam FOV Calibration Tool")
    print("---------------------------")

    # Ask for camera ID
    try:
        camera_id = int(input("Enter camera ID (usually 0 for built-in webcam): "))
    except:
        camera_id = 0
        print("Using default camera ID: 0")

    # Ask for number of images
    try:
        num_images = int(input("Number of calibration images to capture (recommended: 15-20): "))
    except:
        num_images = 15
        print("Using default: 15 images")

    # Capture calibration images
    success = calibrator.capture_calibration_images(camera_id, num_images)

    if not success:
        print("[ERROR] Failed to capture enough calibration images")
        return

    # Perform calibration
    if calibrator.calibrate():
        calibrator.print_results()

        # Test undistortion (optional)
        print("\n[INFO] Press any key to see undistortion effect (or ESC to skip)...")

        cap = cv2.VideoCapture(camera_id)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Undistort the image
            undistorted = cv2.undistort(
                frame,
                calibrator.camera_matrix,
                calibrator.dist_coeffs
            )

            # Show side by side
            combined = np.hstack([frame, undistorted])
            cv2.putText(combined, "Original", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(combined, "Undistorted", (frame.shape[1] + 10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Undistortion Test', combined)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

        cap.release()
        cv2.destroyAllWindows()

    print("\n[INFO] Calibration complete!")


if __name__ == "__main__":
    main()