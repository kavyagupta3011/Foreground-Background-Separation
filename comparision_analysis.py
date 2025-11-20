import matplotlib.pyplot as plt
import json
import os

def plot_tradeoffs():
    file_path = "comparison_data.json"
    
    # 1. Load Data automatically
    if not os.path.exists(file_path):
        print(f"Error: '{file_path}' not found. Run your Median and RPCA algorithms first!")
        return

    with open(file_path, "r") as f:
        data = json.load(f)
    
    # Check if we have the data we need
    required_keys = ["Median Filter (Easy)", "RPCA (Hard)"]
    for key in required_keys:
        if key not in data:
            print(f"Error: Missing data for '{key}'. Please run that algorithm again.")
            return

    # Extract values
    algorithms = required_keys
    f1_scores = [data[algo]["f1"] for algo in algorithms]
    exec_times = [data[algo]["time"] for algo in algorithms]

    # 2. CALCULATE PRICE OF REALISM
    f1_gain = (f1_scores[1] - f1_scores[0]) * 100
    time_cost = (exec_times[1] - exec_times[0])
    price = time_cost / f1_gain if f1_gain > 0 else 0

    print(f" ANALYSIS ")
    print(f"Loaded data from: {file_path}")
    print(f"Gain in Accuracy: +{f1_gain:.2f}%")
    print(f"Cost in Time:     +{time_cost:.2f} seconds")
    print(f"Price of Realism: {price:.2f} seconds per 1% accuracy gain")

    # 3. PLOT TRADEOFF
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Algorithm')
    ax1.set_ylabel('F1-Score', color=color)
    ax1.bar(algorithms, f1_scores, color=color, alpha=0.6, width=0.5)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 1.0)

    # Add text labels on bars
    for i, v in enumerate(f1_scores):
        ax1.text(i, v + 0.02, f"{v:.2f}", ha='center', color='blue', fontweight='bold')

    ax2 = ax1.twinx() 
    color = 'tab:red'
    ax2.set_ylabel('Execution Time (s) - Log Scale', color=color)
    ax2.plot(algorithms, exec_times, color=color, marker='o', linewidth=3, markersize=10)
    ax2.set_yscale('log')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Add text labels for time
    for i, v in enumerate(exec_times):
        ax2.text(i, v * 1.1, f"{v:.1f}s", ha='center', color='red', fontweight='bold')

    plt.title('Trade-off Analysis: Easy vs. Hard Problems')
    plt.tight_layout()
    plt.savefig('tradeoff_chart.png')
    print("Saved tradeoff_chart.png")

if __name__ == "__main__":
    plot_tradeoffs()






