# Foreground–Background Separation using Optimization

## Motivation
- **Real World Challenge:** Reliable separation of moving foregrounds from static backgrounds is critical for applications like video surveillance, robotics, and traffic monitoring.
- **Need for Automation:** Since manually processing millions of pixels is impractical, automated algorithms are essential for efficient object tracking and real-time analysis.
- **Limitations of Naive Methods:** Simple techniques like subtraction or thresholding often fail in dynamic environments due to noise, fast motion, and lighting variations.
- **The Optimization Advantage:** Optimization provides a structured mathematical framework to extract clean foregrounds and preserve background structure, ensuring robustness where traditional methods fall short.

## Dataset
- **Source:** [https://www.kaggle.com/datasets/maamri95/cdnet2014](https://www.kaggle.com/datasets/maamri95/cdnet2014)
- **Purpose:** Designed for benchmarking algorithms in foreground–background separation and motion detection.
- **Diverse Environments:** The dataset encompasses video sequences from highly varied dynamic environments, including crowded areas, highways, and natural scenes.
- **Real-World Complexity:** It explicitly captures difficult real-world variations such as moving vehicles, hard shadows, and camera jitter to rigorously test algorithm robustness.
- **Size & Scale:** The dataset consists of 11 distinct video categories, each category contains 4 - 6 video sequences with over 1,500 frames each.
- **Resolution:** Frame dimensions vary between 320×240 and 720×486 pixels.
- **Key Variables:** The dataset provides raw input video frames alongside precise ground truth masks (per-pixel labels) for validation.

