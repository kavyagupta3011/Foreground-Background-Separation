import os
import sys
import time
import frame_preprocessor

# Import algorithm 
try:
    import run_rcpa_algo as run_rcpa_algo
except ImportError:
    print("Error: Could not import 'rpca_algorithm.py'")
    sys.exit()

# RPCA HYPERPARAMETERS
# Edit these values to change the algorithm's behavior
RPCA_HYPERPARAMS = {
    # only parameter that changes the visual result
    "lambda": None,  # penalty for being foreground.. high penalty means only strictly foreground objects will be counted in foreground.(low recall, high precision)
    
    # tune if slow or o/p unfinished 
    "max_iter": 1000, # reduce to get more speed, if blurry then increase as not converging
    # stopping criteria 
    # for faster less perfect result.. increase value( loosen tolerance ) 
    "eps_primal": 1e-7,
    "eps_dual": 1e-5,

    # admm core 
    "mu": None,
    "rho": 1.6,
    "initial_sv": 10,
    "max_mu": 1e6,

    # utility parameters
    "verbose": True,      # false when done tuning 
    "save_interval": None # for debugging if needed 
}

# The script will build a path like:
# ../dataset / [CategoryName] / [VideoName] / input
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

    print("Foreground - Background Seperation ( Hard Problem - RPCA )")
    
    chosen_algo = "RPCA"
    
    # Select Dataset Category
    categories = list(DATASET_OPTIONS.keys())
    chosen_category = select_from_list(categories, "Select Dataset Category")
    
    # Select Specific Video
    video_list = DATASET_OPTIONS[chosen_category]
    chosen_video = select_from_list(video_list, f"Select Video in '{chosen_category}'")
    
    # Generate the final path
    final_path = os.path.join(DATASET_ROOT, chosen_category, chosen_video, "input")
    
    print()
    print("Configuration Complete:")
    print(f"  Algorithm: {chosen_algo}")
    print(f"  Dataset:   {chosen_category} / {chosen_video}")
    print(f"  Input Path: {final_path}")
    print()
   
    run_rcpa_algo.run_rpca(final_path, chosen_video, RPCA_HYPERPARAMS)

    print("\nProcess finished. Exiting...")

if __name__ == "__main__":
    main()


# How to solve using RPCA:
# We have a hard problem. We want to find simple background and sparse foreground with lowest possible objective cost, obeying L + S = M 
# Augmented Lagrangian: 
#           trick to solve constrained problem.
#           instead of minimizing score while obeying rule, create a new unconstrained problem 
# Create penalty function that includes:
# The Original Score: ||L||* + \lambda||S||1 (Background Simplicity + Foreground Sparsity)
# A "Referee" (Y): This is the Lagrange Multiplier. It's a matrix that tracks the error (M - L - S).
# A "Penalty" (mu): This is a number that controls how much the algo get punished for having any error.
# The new "Total Score" becomes a big mix of all three. Now, the algorithm's only goal is to minimize this new score, and by doing so, 
# it will automatically be forced to make the error ($M - L - S$) go to zero.

# Inexact ALM 
# Solving for both L and S at the same time is too hard, we solve one at a time
# Step 1: if I freeze the background (L) and the referee (Y), what is the best possible foreground (S) (clean up the mess left by the other two)
# Step 2: Now that I have a new S, I'll freeze it. What is the best possible background (L)? (Find simplest background that explains the rest of the mess)
# Step 3: Ok how big is our error now(M-L-S)? tells what mistakes to fix in the next iteration
# Step 4: Getting closer, increase penalty so that error is punished even more. Focus very precisely on getting primal error to zero
# repeat 1-4; till errors are effectively zero 