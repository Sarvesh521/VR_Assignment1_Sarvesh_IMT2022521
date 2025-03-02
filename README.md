# VR_Assignment1_Sarvesh_IMT2022521

This repository contains a computer vision assignment implemented in Python. It covers coin detection, segmentation, coin counting, and panorama creation. Detailed instructions, methodology, results, and observations are provided below. The solution is available as both Python scripts and Jupyter notebooks.

---

## Table of Contents

- [Running the Code](#running-the-code)
- [Project Documentation](#project-documentation)
- [Part A: Coin Detection, Segmentation & Counting](#part-a-coin-detection-segmentation--counting)
- [Part B: Panorama Creation](#part-b-panorama-creation)
- [Final Report and Conclusions](#final-report-and-conclusions)

---

## Running the Code

The code can be executed via the command line or through Jupyter notebooks.

- **Via Command Line:**

  - For coin detection, segmentation, and counting:
    ```sh
    python Coins.py
    ```
  - For panorama creation:
    ```sh
    python Panorama.py
    ```

- **Via Jupyter Notebooks:**

  - Open and run the individual notebooks:
    - [Part1-coins.ipynb](http://_vscodecontentref_/0)
    - [Part2-Panorama.ipynb](http://_vscodecontentref_/1)

---

## Project Documentation

This repository is self-contained and runs without additional intervention. All dependencies are listed in the [requirements.txt](http://_vscodecontentref_/2) file. Detailed documentation is provided below, and all visual outputs—such as detection, segmentation, coin counts, and panoramas—are clearly labeled.

---

## Part A: Coin Detection, Segmentation & Counting

### Input Data

- Input images for coin processing are located in the [Part-1](http://_vscodecontentref_/3) directory, including files like `img1.jpg`, `img2.jpg`, and `img3.jpg`.

### Processing and Methods

- **Detection:**  
  Coin boundaries are identified using advanced edge detection techniques.
  
- **Segmentation:**  
  The detected edges allow for effective segmentation of coins from the background.
  
- **Counting:**  
  Each segmented coin is counted, and the total is annotated on the processed images.

### Output

- Results are saved under the [Part1](http://_vscodecontentref_/4) directory:
  - Images highlight detected coin contours.
  - Segmented coin images are provided.
  - Overlays display the coin count on the images.

### Observations

- A comparative analysis is included regarding different edge detection methods.
- The effectiveness of segmentation and counting is discussed, and visual outputs help in verifying the results.

---

## Part B: Panorama Creation

### Input Data

- The images used for creating panoramas are stored in the [Part-2](http://_vscodecontentref_/5) directory.

### Processing and Methods

- **Feature Detection & Matching:**  
  Techniques like SIFT and ORB are used to identify and match keypoints across images.
  
- **Image Stitching:**  
  Two stitching approaches are applied:
  - **Basic Stitching:** Direct stitching without blending.
  - **Feather Blending:** Creates smoother transitions between images.
  
- **Cropping:**  
  The stitched panorama is cropped to remove unwanted borders.

### Output

- The final panoramic images are stored in the [Part2](http://_vscodecontentref_/6) directory:
  - Displays include keypoint visualizations.
  - Both basic and feather-blended panoramas are generated.

### Observations

- Detailed analysis is provided for keypoint matching and panorama quality.
- Visual comparisons demonstrate the benefits of feather blending over the basic method.

---

## Final Report and Conclusions

### Approaches and Methodology

- **Coin Detection & Segmentation:**  
  Several edge detection and segmentation methodologies were tested and refined to achieve optimal coin extraction.
  
- **Panorama Creation:**  
  Multiple panorama stitching techniques were evaluated. Feather blending emerged as a superior method for creating seamless panoramas.

### Results

- Both script-based and notebook-based implementations run autonomously without further intervention.
- The performance of each module is validated by the accuracy of coin counts and the visual quality of panoramas.
  
### Lessons Learned

- **Successes:**  
  Refined segmentation and effective panorama blending were key achievements.
  
- **Challenges:**  
  Fine-tuning parameters for various methods required iterative testing, and handling a range of image qualities posed challenges.
  
- **Final Approach:**  
  The implementations in [Coins.py](http://_vscodecontentref_/7) and [Panorama.py](http://_vscodecontentref_/8) represent balanced solutions for stability and performance. Jupyter notebooks further facilitate interactive testing and validation.

---

This README file provides comprehensive guidance on how to run the project, understand the processing methods employed, and review the results. All visual outputs are well documented, ensuring clarity in the process and outcomes.

For further details, please refer to inline comments in the source code and the accompanying Jupyter notebooks.