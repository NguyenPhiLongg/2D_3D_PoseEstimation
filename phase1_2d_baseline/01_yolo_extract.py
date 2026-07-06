import os
import cv2
import torch
import pandas as pd
from ultralytics import YOLO

# Configure paths
INPUT_DIR = "data/1_raw_videos"
OUTPUT_DIR = "data/2_extracted_2d"
ERROR_LOG_PATH = os.path.join(OUTPUT_DIR, "error_videos.txt")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_2d_pose():
    # Check GPU
    if torch.cuda.is_available():
        print("GPU check: CUDA is available")
        DEVICE = 0   # Use GPU
    else:
        print("GPU check: CUDA is not available, using CPU")
        DEVICE = "cpu"

    # Initialize YOLOv8 Pose model
    model = YOLO("yolov8m-pose.pt")

    # Scan all videos in the input folder and its subfolders
    video_files = []

    for root, dirs, files in os.walk(INPUT_DIR):
        for file in files:
            if file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                full_path = os.path.join(root, file)
                video_files.append(full_path)

    if not video_files:
        print("No videos found in data/1_raw_videos/ or its subfolders")
        return

    print(f"Found {len(video_files)} video(s).")

    skipped_count = 0
    processed_count = 0
    no_person_count = 0
    error_videos = []

    for video_path in video_files:
        video_name = os.path.basename(video_path)

        # Keep folder structure information in output file name
        relative_path = os.path.relpath(video_path, INPUT_DIR)
        relative_name = relative_path.replace("\\", "_").replace("/", "_")
        output_csv_name = os.path.splitext(relative_name)[0] + ".csv"
        output_csv_path = os.path.join(OUTPUT_DIR, output_csv_name)

        # Skip video if CSV already exists
        if os.path.exists(output_csv_path):
            skipped_count += 1
            continue

        print(f"\nProcessing: {video_path}")

        cap = None

        try:
            # Open video
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print(f"WARNING: Cannot open video: {video_path}")
                error_videos.append(video_path)
                continue

            all_frames_data = []
            frame_idx = 0

            while cap.isOpened():
                success, frame = cap.read()

                if not success:
                    break

                try:
                    # Run YOLO pose detection using GPU
                    results = model(frame, verbose=False, device=DEVICE)

                except Exception as e:
                    print(f"ERROR: YOLO failed at frame {frame_idx} in video {video_name}")
                    print(f"Error detail: {e}")
                    error_videos.append(video_path)
                    break

                # Extract keypoints if a person is detected
                for result in results:
                    if result.keypoints is not None and len(result.keypoints.xy) > 0:
                        # Get 17 COCO keypoints from the first detected person
                        keypoints = result.keypoints.xy[0].cpu().numpy()

                        # Flatten 17x2 keypoints into 34 values
                        flattened_kpts = keypoints.flatten()

                        # Save current frame data
                        row_data = [frame_idx] + flattened_kpts.tolist()
                        all_frames_data.append(row_data)

                frame_idx += 1

                # Print progress every 30 frames
                if frame_idx % 30 == 0:
                    print(f"Processed {frame_idx} frames in {video_name}")

            # Release video
            cap.release()
            cap = None

            # Create column names
            columns = ["frame"]
            for i in range(17):
                columns.extend([f"x{i}", f"y{i}"])

            # Save to CSV
            if all_frames_data:
                df = pd.DataFrame(all_frames_data, columns=columns)
                df.to_csv(output_csv_path, index=False)
                processed_count += 1
                print(f"Successfully saved: {output_csv_path} ({len(df)} detected frames)")
            else:
                no_person_count += 1
                print(f"WARNING: No person was detected in video {video_path}")

        except KeyboardInterrupt:
            print("\nProgram stopped by user using Ctrl + C.")
            print("CSV files that were already saved are still safe.")
            break

        except Exception as e:
            print(f"ERROR while processing video: {video_path}")
            print(f"Error detail: {e}")
            error_videos.append(video_path)

        finally:
            if cap is not None:
                cap.release()

    # Save error video list
    if error_videos:
        with open(ERROR_LOG_PATH, "w", encoding="utf-8") as f:
            for video in error_videos:
                f.write(video + "\n")

    # Final summary
    print("\nDone.")
    print(f"Total videos found: {len(video_files)}")
    print(f"Skipped existing CSV files: {skipped_count}")
    print(f"Processed new videos: {processed_count}")
    print(f"Videos with no detected person: {no_person_count}")
    print(f"Error videos: {len(error_videos)}")

    if error_videos:
        print(f"Saved error video list to: {ERROR_LOG_PATH}")


if __name__ == "__main__":
    extract_2d_pose()