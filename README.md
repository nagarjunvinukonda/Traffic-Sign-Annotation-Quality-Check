# Scale AI Traffic Sign Annotation Quality-Check Framework

This repository contains a Python-based quality-check framework for Scale AI traffic sign annotation tasks, along with a Phase 3 reflection documenting the work. It took me 4.5 hours in total to analyse, code and document my results.

---

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Quality Checks](#quality-checks)
- [Results Summary](#results-summary)
- [Future Directions & Enhancements](#future-directions--enhancements)
- [Next Steps](#next-steps)
- [References](#references)

---

## Introduction

This project implements `quality_checks.py`, a script that:

- Queries annotation tasks from the Scale API using `ScaleClient`.
- Downloads images once per task and performs a suite of programmatic validation checks on each bounding box annotation.
- Supports automatic sharding of output files for large-scale datasets.
- Outputs results in CSV files, flagging errors and warnings to streamline human review.

The Phase 3 reflection (included below) covers:

- The checks implemented and their severities.
- A summary of results for the sample 8 tasks.
- High-impact ideas for future improvements.

---

## Installation

Clone the repository:

```sh
git clone <repo-url>
cd <repo-dir>
```

Install Python and pip along with it:
- [Follow the download link for Python installation on Windows](https://www.python.org/downloads/)


Create a Python environment & install dependencies:

```sh
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Set your Scale API key:

```sh
export SCALE_API_KEY="live_ebacb4d3144e445a8af4547603b2b78e"
```

---

## Usage

For our project the `tasks.json` is already populated with the array of Scale task IDs as follows:

```json
{
  "tasks": [
    "5f127f6f26831d0010e985e5",
    "5f127f6c3a6b1000172320ad",
    "..."
  ]
}
```

Run the quality checks with automatic sharding at 50,000 tasks per file (This will be useful for large dataset):
- **Note**: I am caching the image per task avoid re-downloading the full image for every quality check and analysis. 
```sh
python quality_checks.py
```

Review the generated CSV files:

- `results_1-50000.csv`
- `results_50001-100000.csv`
- …

The data sharding happens as above to take input of large dataset and shard it's results into different .csv files.  

---

## Quality Checks

### **Bounds (Error):** 
- Verifies bounding boxes lie entirely within image dimensions. Ensures left ≥ 0, top ≥ 0, left + width ≤ img_width, top + height ≤ img_height.
- I want to ensure the bounding boxes corners are not outside the image itself.
### **MinSize (Warning):** 
- Flags boxes with width or height < 10px.  
- Its hard to label smaller bounding boxes and there is a high chance I might make some mistakes. This Warning ensure to recheck again.   
### **AspectRatio (Warning):** 
- Warns on boxes with extreme ratio (w/h > 5 or < 0.2).
- There are some bounding boxes which are either too tall or too wide, where there is chance you are overlapping or bounding box creation might be wrong.  
### **AtLeastOneTrafficControl (Warning):** 
- Confirms each image has at least one `traffic_control_sign`.
- The dataset I am perfroming quality checks is for Traffic Signs dataset which should consist of atleast one traffic sign else there is an error in our task.  
### **TrafficLightBackground (Error):** 
- Uses Hough circle detection + HSV masks to identify real traffic lights and enforce `background_color = "other"`.
- As provided in the Traffic Sign Spec Document, common errors section I am searching for traffic light and ensuring it's background color is other. The current methods have a few drawbacks which is explained later below on how to improve better model for future purposes.  
### **OverlapDuplicates (Warning):** 
- Flags pairs of boxes with Intersection-over-Union (IoU) > 0.5.
- There are a few tasks which have duplicate bouding boxes and labels for the same single object.
### **WholeImage (Error):** 
- Catches any box that spans the entire image frame.
- I dont want to have any bounding box spanning the entire image.
---

## Results Summary

A sample run on the 8 provided tasks yielded these illustrative totals (see the CSVs for exact flags):

| Task ID                  | Errors | Warnings |
|--------------------------|--------|----------|
| 5f127f6f26831d0010e985e5 |   0    |    4     |
| 5f127f6c3a6b1000172320ad |   0    |    2     |
| 5f127f699740b80017f9b170 |   0    |    0     |
| 5f127f671ab28b001762c204 |   0    |    18    |
| 5f127f643a6b1000172320a5 |   0    |    7     |
| 5f127f5f3a6b100017232099 |   0    |    6     |
| 5f127f5ab1cb1300109e4ffc |   1    |    2     |
| 5f127f55fdc4150010e37244 |   0    |    7     |
| **Total**                |   1    |    46    |

Refer to `results_1-...csv` files for detailed per-UUID issue lists.

---

## Future Directions & Enhancements on current model:


### Furture changes to be added for Traffic light detection i.e. TrafficLightBackground (Error): 

- **SignBackgroundMismatch (Error):** 
- Without Neural Networks: For large color masks (>30×30px), I classify static signs and check HSV-derived background color against labels. This helps to distinguish between traffic lights and signs with similar color intensity. 
- Example: With the Hough circle detection + HSV masks I identify real traffic lights but if there is a traffic Sign with same intensity as a traffic signal, this method fails. If I derived a statistical threshold of no.of pixels for the traffic signal circles per task I can differentaite between traffic sign and traffic signal as the traffic sign has huge no.of pixels comparitively. 

- **ML-Backed Validation:** Integrate a lightweight classification or object detection model (e.g. a small YOLO variant) trained specifically on sign vs. light vs. non‐sign objects. This would reduce brittle HSV thresholds and catch unusual cases (e.g. novelty signs, low light conditions). This will help specifically for task: "5f127f699740b80017f9b170". 


### Improvments for AspectRatio and MinSize:
- **Dynamic Threshold Calibration:** – Rather than hard-coding values like `MinSize=10px` or `IoU>0.5`, **Dynamic Threshold Calibration** automatically derives sensible cutoffs from actual data distributions. Hence, collect histograms of IoU, aspect ratios, and size distributions across a larger dataset to automatically set dynamic thresholds (e.g. MinSize as 1% of image area, IoU cutoff tuned per label).

---


## Next Steps: 

**Real-Time Annotation Feedback:** Embed checks within the annotation interface to prevent errors before submission.

In our initial Phase 2 & 3 work I have covered core geometric and color–shape checks. With more time, I would next tackle several unaddressed rules from the Traffic Sign Spec Document :

### 1. **Free-standing Signs Only**  
**Spec:** “Signs must be free-standing; Signs painted on buildings or the road should not be annotated.”  
**Why it matters:** Painted or wall-mounted signs are out of scope and inflate annotation noise.  
**Proposed approaches:**
- **Semantic Segmentation / Depth:** Use a lightweight “road + building vs. pole” segmentation model (e.g. a U-Net variant) or monocular depth estimation to verify that each box sits on a pole/stand rather than directly on a planar wall.  
- **Pole Detection Heuristic:** Detect vertical line segments (Hough lines) under the sign crop; if no vertical support is found, flag as “NonFreeStanding.”  
- **ML Classifier:** Train a small CNN on “free-standing vs. painted” sign crops to catch corner cases (e.g. murals).

### 2. **Exclude Commercial & Event Signs**  
**Spec:** “Signs for commercial activities or events should not be annotated.”  
**Why it matters:** I only want traffic-related signage, not store adverts or event banners.  
**Proposed approaches:**
- **OCR + Keyword Filter:** Run Tesseract or a lightweight text-detector on the crop; if extracted text matches a commercial/event lexicon (e.g. “Sale”, “Open”, “Festival”), flag as “CommercialSign.”  
- **Image-based Classifier:** Fine-tune a ResNet-style classifier to distinguish official traffic shapes vs. stylized marketing layouts.  
- **Metadata Tagging:** If the original image URL or EXIF contains GPS/location metadata outside public roads (e.g. inside malls), deprioritize those tasks.

### 3. **Individual Sign Labeling (No Group Boxes)**  
**Spec:** “Signs should be individually labeled; Do not label signs in groups.”  
**Why it matters:** One box per sign instance improves downstream model training granularity.  
**Proposed approaches:**
- **Connected-Component Analysis:** Within each crop, convert to binary via edge detection or color thresholds; count distinct connected components (e.g. multiple sign shapes). If >1, flag “GroupedSigns.”  
- **Contour/Hough-Shape Clustering:** Detect separate circular/rectangular contours—if more than one contour of the correct sign shape exists, it indicates multiple signs.  
- **IoU Clustering:** If an annotation’s box significantly overlaps two or more other sign centroids (via precomputed sign-centroid map), treat as a grouped label.

---

## References

- [Scale AI Docs: Overview](https://scale.com/docs)
- [Scale API Refernces: Tasks Retrive](https://scale.com/docs/api-reference/tasks#retrieve-a-task)

