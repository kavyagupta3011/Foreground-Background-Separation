import shutil       # To delete temp files
import cv2          # For reading images
import numpy as np  # For stacking frames
import os           # For building file paths
import glob         # To find all the image files
import time         # To track execution time
import json         # To store comparision data
import matplotlib.pyplot as plt 
import frame_preprocessor

def create_difference_map(generated_mask, gt_mask):
    """
    Creates a color-coded map showing evaluation results.
    Green = True Positive (Found)
    Red   = False Positive (Noise/Ghost)
    Blue  = False Negative (Missed)
    """
    h, w = generated_mask.shape
    diff_map = np.zeros((h, w, 3), dtype=np.uint8)

    # Define boolean masks
    my_FG = (generated_mask > 127)
    gt_FG = (gt_mask == 255)
    gt_BG = (gt_mask == 0) | (gt_mask == 170)

    # Green: True Positive (Intersection)
    diff_map[my_FG & gt_FG] = [0, 255, 0] 

    # Red: False Positive (Noise)
    diff_map[my_FG & gt_BG] = [0, 0, 255]

    # Blue: False Negative (Missed)
    diff_map[~my_FG & gt_FG] = [255, 0, 0]
    
    return diff_map

def save_maps_to_video(map_list, output_path, fps=30):
    """
    Saves a list of color images to a video file. (For difference maps)
    """
    if not map_list:
        return

    height, width, layers = map_list[0].shape
    size = (width, height)
    
    # Create video writer
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    
    for i in range(len(map_list)):
        out.write(map_list[i])
    out.release()
    print(f"Saved evaluation video to: {output_path}")

def evaluate_performance(generated_folder, ground_truth_folder, chosen_video, algorithm_name):
    """
    Compares generated binary masks against CDnet ground truth.
    """
    print(f"Generated masks: {generated_folder}")
    print(f"Ground Truth:    {ground_truth_folder}")

    # Setup output folder
    map_output_folder = f"./evaluation_results/{algorithm_name}/{chosen_video}"
    if os.path.exists(map_output_folder):
        shutil.rmtree(map_output_folder)
    os.makedirs(map_output_folder, exist_ok=True)
    
    difference_frames = []

    # Load Files
    gen_files = sorted(glob.glob(os.path.join(generated_folder, "*.png")))
    all_gt_files = sorted(glob.glob(os.path.join(ground_truth_folder, "*.png")))
    
    if not all_gt_files:
        all_gt_files = sorted(glob.glob(os.path.join(ground_truth_folder, "*.jpg")))

    if not all_gt_files:
        print("Error: No ground truth files found.")
        return 0.0, 0.0

    # Apply slicing to get the frames used 
    result = frame_preprocessor.preprocess_frames(all_gt_files, chosen_video)
    if isinstance(result, tuple):
        gt_files = result[0] 
    else:
        gt_files = result
        
    # Initialize Counters
    TP, FP, FN = 0, 0, 0
    limit = min(len(gen_files), len(gt_files))
    print(f"Comparing {limit} aligned frames...")

    for i in range(limit):
        # Read Generated Mask
        gen = cv2.imread(gen_files[i], cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(gt_files[i], cv2.IMREAD_GRAYSCALE)

        if gen is None or gt is None:
            continue
            
        if gen.shape != gt.shape:
            gen = cv2.resize(gen, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

        # CDnet Logic 
        gt_FG = (gt == 255)
        gt_BG = (gt == 0) | (gt == 170) 
        my_FG = (gen > 127)
        my_BG = (gen <= 127)

        # Metrics
        TP += np.sum(my_FG & gt_FG)
        FP += np.sum(my_FG & gt_BG)
        FN += np.sum(my_BG & gt_FG)
        
        # Create map with legend 
        diff_map = create_difference_map(gen, gt)
        cv2.putText(diff_map, f"Algo: {algorithm_name}", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(diff_map, "Green: Correct", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(diff_map, "Red: Noise", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(diff_map, "Blue: Missed", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        difference_frames.append(diff_map)

    # Save the map as video
    video_filename = os.path.join(map_output_folder, f"{chosen_video}_eval_map.mp4")
    save_maps_to_video(difference_frames, video_filename, fps=10)

    # Compute Final Scores
    iou = TP / (TP + FP + FN + 1e-6)    
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)

    print("-" * 30)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1_score:.4f}")
    print(f"IoU:       {iou:.4f}")
    print("-" * 30)

    return f1_score, iou

def log_results_to_file(algorithm_name, f1_score, iou_score, execution_time):
    """
    Saves the F1 score, IoU and Time to a JSON file for later comparison.
    """
    file_path = "../comparison_data.json"
    
    # Load existing data if it exists
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
    else:
        data = {}
        
    # Update the data for this specific algorithm
    data[algorithm_name] = {
        "f1": f1_score,
        "iou": iou_score,
        "time": execution_time
    }
    
    # Save it back
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
    
    print(f"Results for '{algorithm_name}' saved to {file_path}")