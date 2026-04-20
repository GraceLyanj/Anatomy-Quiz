# Anatomy Teaching App 🫀🧠

Interactive medical image anatomy quiz using SAM (Segment Anything Model) segmentation with real anatomical labels.

## Features
- ✅ Real anatomical organ labels from BTCV dataset (13 organs)
- ✅ Interactive segmentation using Meta's SAM model
- ✅ Dice score-based grading (compares user selection to ground truth)
- ✅ 178 training images of abdominal CT scans
- ✅ Visual feedback showing correct answers

## Organs Included
1. Spleen
2. Right Kidney
3. Left Kidney
4. Gallbladder
5. Esophagus
6. Liver
7. Stomach
8. Aorta
9. Inferior Vena Cava
10. Portal/Splenic Vein
11. Pancreas
12. Right Adrenal Gland
13. Left Adrenal Gland

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- ~2GB free disk space for model weights

### Installation

1. **Clone this repository:**
   ```bash
   git clone https://github.com/zzegub-svg/anatomy-teacher-app.git
   cd anatomy-teacher-app
   
Install required packages:

bash
pip install segment-anything opencv-python matplotlib numpy pillow torch torchvision scipy
Download SAM model weights (375MB):

Download from: https://drive.google.com/file/d/1X2UwYpfSpw8uPHPkhBIsRQKPFViqhkpV/view?usp=sharing
Place the file in: ckpt/sam_vit_b_01ec64.pth
Get BTCV dataset:

Clone IMIS-Bench: git clone https://github.com/uni-medical/IMIS-Bench.git
Place it in the parent directory (one level up from this project)
Run the app:

bash
python anatomy_teacher_2.py

## How to Use

1. **Read the question** at the top (e.g., "Click on the Liver")
2. **Click on the organ** in the CT scan image
3. **See your selection** highlighted in green
4. **Click "Submit Answer"** to check if you're correct
5. **View feedback:**
   - Green = your selection
   - Blue = correct answer
   - Dice score shows accuracy (70%+ = excellent!)
6. **Click "Next Question"** to continue

## Grading System
- **Excellent (70%+ accuracy):** +10 points
- **Good (50-70% accuracy):** +5 points
- **Partial (20-50% accuracy):** 0 points, try again
- **Incorrect (<20% accuracy):** 0 points, hint provided

## Technical Details
- **Segmentation Model:** Meta's Segment Anything Model (SAM) - ViT-B
- **Dataset:** BTCV (Beyond The Cranial Vault) - 13 abdominal organs
- **Evaluation Metric:** Dice Similarity Coefficient
- **UI Framework:** Tkinter

## Credits
Based on research from:
- Segment Anything Model (SAM) - Meta AI
- IMIS-Bench - Interactive Medical Image Segmentation Benchmark
- BTCV Dataset - Multi-organ segmentation challenge

## Authors
- Zachary Zegub
- Grace Liu

University of Texas at Austin - Computer Vision Project
