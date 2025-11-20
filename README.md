# Foreground–Background Separation using Optimization

###  AI 503 – Optimization Project

##  Team Members

*  **Kavya Gupta (IMT2023016)**
*  **Nainika Agrawal (IMT2023034)**
*  **S. Divya (IMT2023059)**

---

# Dataset

We used a video-based dataset for motion-based foreground–background separation.

 **CDNet 2014 Dataset (Kaggle):**
[https://www.kaggle.com/datasets/maamri95/cdnet2014](https://www.kaggle.com/datasets/maamri95/cdnet2014)

---

#  Required Imports

```python
import shutil       # To delete temp files
import cv2          # For reading images
import numpy as np  # For stacking frames
import os           # For building file paths
import glob         # To find all the image files
import time         # To track execution time
import matplotlib.pyplot as plt
import frame_preprocessor

import os
import sys
import time
```

---

#  Project Overview

We perform **foreground–background separation** on video sequences using:

# Easy Problem — Mean & Median Filtering

##  Idea

We estimate the background by updating pixel statistics across frames.

Two filtering strategies:

### **1️⃣ Running Mean Filter**

Updates background using an exponential moving average:
[
B_t = \alpha I_t + (1 - \alpha) B_{t-1}
]

### **2️⃣ Running Median (Approx.)**

A lightweight approximation of median behavior through incremental updates.

This background is then compared with the current frame to extract the foreground.

---

## ⏱ Time Complexity

* Updating each pixel: **O(1)**
* For **P pixels per frame** → per-frame cost: **O(P)**
* For **N frames** → total cost:
  [
  \boxed{O(NP)}
  ]

This is extremely fast and scalable.

---

#  Hard Problem — Robust PCA (RPCA)

##  Idea

We reshape frames into column vectors to form a data matrix ( M ), then solve:

[
M = L + S
]

Where:

* **L:** Low-rank matrix → background
* **S:** Sparse matrix → moving objects

This is solved using optimization techniques such as **Principal Component Pursuit (PCP)**, which relies heavily on repeated **SVD (Singular Value Decomposition)**.

---

## ⏱ Time Complexity

* SVD of a matrix with ( P ) pixels and rank ( k ):
  [
  O(Pk)
  ]
* If RPCA runs for **I iterations**, total complexity becomes:

[
\boxed{O(I \cdot N \cdot P \cdot k)}
]

This is **much slower** than simple filtering but yields **significantly better accuracy** for complex scenes.

---

