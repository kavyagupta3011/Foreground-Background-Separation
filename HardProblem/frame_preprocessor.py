
# This file contains the frame preprocessing rules for each video dataset.
# The rule is a tuple: (START, STOP, STEP)
PREPROCESSING_RULES = {
    # --- baseline ---
    "PETS2006": ( 300, 1200, 2),
    "highway": ( 470, 1700, 2), #
    "office": ( 570, 2050, 2),
    "pedestrians": ( 300, 1099, 2), #
    
    # --- badWeather ---
    "blizzard": ( 900, 7000, 2),
    "skating": ( 800, 3900, 2), #
    "snowFall": ( 800, 6500, 2),
    "wetSnow": ( 500, 3500, 2), 
    
    # --- cameraJitter ---
    "badminton": ( 800, 1150, 2), #
    "boulevard": ( 790, 2500, 2),
    "sidewalk": ( 800, 1200, 2), #
    "traffic": ( 900, 1570, 2), 
    
    # --- dynamicBackground ---
    "boats": ( 1900, 7999, 3),
    "canoe": ( 800, 1189, 2),
    "fall": ( 1000, 4000, 2),
    "fountain01": ( 400, 1184, 2), #
    "fountain02": ( 500, 1499, 2), #
    "overpass": ( 1000, 3000, 2),
    
    # --- intermittentObjectMotion ---
    "abandonedBox": ( 2450, 4500, 2),
    "parking": ( 1100, 2500, 2), #
    "sofa": ( 500, 2750, 2),
    "streetLight": ( 175, 3200, 2),
    "tramstop": ( 1320, 3200, 2),
    "winterDriveway": ( 1000, 2500, 2), #
    
    # --- lowFramerate ---
    "port_0_17fps": ( 1000, 3000, 1),
    "tramCrossroad_1fps": ( 400, 900, 1), #
    "tunnelExit_0_35fps": ( 2000, 4000, 1),
    "turnpike_0_5fps": ( 800, 1500, 1), #
     
    # --- nightVideos ---
    "bridgeEntry": ( 1000, 2500, 2),
    "busyBoulvard": ( 730, 2760, 2),
    "fluidHighway": ( 400, 1364, 2), #
    "streetCornerAtNight": ( 800, 5200, 2),
    "tramStation": ( 500, 3000, 2),
    "winterStreet": ( 900, 1785, 2), #
    
    # --- PTZ ---
    "continuousPan": (600, 1700, 2), #
    "intermittentPan": (1200, 3500, 2),
    "twoPositionPTZCam": (800, 2300, 2),
    "zoomInZoomOut": (500, 1130, 2), #
    
    # --- shadow ---
    "backdoor": (400, 2000, 2),
    "bungalows": (300, 1700, 2),
    "busStation": (300, 1250, 2), #
    "copyMachine": (500, 3400, 2),
    "cubicle": (1100, 7400, 2),
    "peopleInShade": (250, 1199, 2), #
    
    # --- thermal ---
    "corridor": (500, 5400, 2),
    "diningRoom": (700, 3700, 2),
    "lakeSide": (1000, 6500, 2),
    "library": (600, 4900, 2),
    "park": (250, 600, 2), #
    
    # --- turbulence ---
    "turbulence0": (1000, 5000, 2),
    "turbulence1": (1200, 4000, 2),
    "turbulence2": (500, 4500, 2),
    "turbulence3": (800, 2200, 2) #
}

def preprocess_frames(all_frame_files, video_name):
    """
    Applies a specific (start, stop, step) slice rule to the list of 
    frame files based on the video name.
    
    Args:
        all_frame_files (list): The full, sorted list of all frame paths.
        video_name (str): The name of the video (e.g., "highway").
        
    Returns:
        tuple: (processed_file_list)
    """
    
    # Default rule: Process ALL frames.
    default_rule = (None, None, 1) 
    
    # Get the rule from the dictionary. If 'video_name' isn't found, use 'default_rule'.
    rule = PREPROCESSING_RULES.get(video_name, default_rule)
    start, stop, step = rule
    processed_files = all_frame_files[start:stop:step]
        
    return processed_files