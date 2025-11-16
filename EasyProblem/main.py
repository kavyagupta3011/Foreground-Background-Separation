import os
import sys
import time

# import algorithm models 
try:
    import mean_filter
    import median_filter
except ImportError:
    print("Error: Could not import algorithm files.")
    sys.exit()


# The script will build a path like: ../dataset / [CategoryName] / [VideoName] / input
DATASET_ROOT = "../dataset"
DATASET_OPTIONS = {
    "baseline": ["highway", "office", "pedestrians", "PETS2006",],
    "badWeather": ["blizzard", "skating", "snowFall", "wetSnow"],
    "cameraJitter": ["badminton", "boulevard", "sidewalk", "traffic"],
    "dynamicBackground": ["boats", "canoe", "fall", "fountain01", "fountain02", "overpass"],
    "intermittentObjectMotion": ["abandonedBox", "parking", "sofa", "streetLight", "tramstop", "winterDriveway"],
    "lowFramerate": ["port_0_17fps", "tramCrossroad_1fps", "tunnelExit_0_35fps", "turnpike_0_5fps"],
    "nightVideos": ["bridgeEntry", "busyBoulvard", "fluidHighway", "streetCornerAtNight", "tramStation", "winterStreet"],
    "PTZ": ["continuousPan", "intermittentPan", "twoPositionPTZCam", "zoomInZoomOut"],
    "shadow": ["backdoor", "bungalows", "busStation", "copyMachine", "cubicle", "peopleInShade"],
    "thermal": ["corridor", "diningRoom", "lakeSide", "library", "park"],
    "turbulence": ["turbulence0", "turbulence1", "turbulence2", "turbulence3"]
}
    

def select_from_list(options, title):
    """
    A helper function to create a reusable command-line menu.
    
    Args:
        options (list): A list of strings to choose from.
        title (str): The title to display for the menu.
        
    Returns:
        str: The selected option.
    """

    while True:
        print(f"\n--- {title} ---")
        for i, option in enumerate(options):
            print(f"  {i+1}. {option}")
        
        choice = input(f"\nEnter your choice (1-{len(options)}): ")
        
        # Validate the input
        if not choice.isdigit():
            print("Invalid input. Please enter a number.")
            continue
            
        # Convert to 0-based index
        choice_idx = int(choice) - 1 
        
        if 0 <= choice_idx < len(options):
            return options[choice_idx] 
        else:
            print(f"Invalid choice. Must be between 1 and {len(options)}.")

def main():
    """
    Main function to run the application.
    """

    print("Foreground - Background Seperation ( Easy Problem )")
    
    # 1. Select Algorithm
    algorithms = ["Mean Filter", "Median Filter"]
    chosen_algo = select_from_list(algorithms, "Select an Algorithm")
    
    # 2. Select Dataset Category
    categories = list(DATASET_OPTIONS.keys())
    chosen_category = select_from_list(categories, "Select Dataset Category")
    
    # 3. Select Specific Video
    video_list = DATASET_OPTIONS[chosen_category]
    chosen_video = select_from_list(video_list, f"Select Video in '{chosen_category}'")
    
    # 4. Generate the final path
    final_path = os.path.join(DATASET_ROOT, chosen_category, chosen_video, "input")
    
    print()
    print("Configuration Complete:")
    print(f"  Algorithm: {chosen_algo}")
    print(f"  Dataset:   {chosen_category} / {chosen_video}")
    print()
    
    # 5. Run the chosen algorithm with the generated path
    if chosen_algo == "Mean Filter":
        mean_filter.run_mean_filter(final_path, chosen_video)
    elif chosen_algo == "Median Filter":
        median_filter.run_median_filter(final_path, chosen_video)
    else:
        print("Error: No function defined for this algorithm.")
       
    print("\n Process finished. Exiting..")

if __name__ == "__main__":
    main()