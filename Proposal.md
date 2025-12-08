
# CSC173 Deep Computer Vision Project Proposal
**Student:** [Joseph Jr. Q Corpuz], [2020-1360]  
**Date:** [12-12-2025]

## 1. Project Title 
[ASL Sign Recognition Using Real-Time Computer Vision]

## 2. Problem Statement
[American Sign Language, like all languages, is difficult to learn. Most ASL learning tools require internet, subcription, and often lack real time visual validation. This project aims to address this by building a accesible ASL alphabet recognizer that provides feedback using standard webcams.]

## 3. Objectives
- Develop and train a lightweight CNN model to classify ASL hand signs for letters A-Z and a "nothing" class
- Achieve At least ≥85% validation accuracy on a test set.
- Implement the pipeline : data loading -> Preprocessing -> Training -> Evaluation -> Real time inference using webcam.

## 4. Dataset Plan
- Source: [https://www.kaggle.com/datasets/grassknoted/asl-alphabet?spm=a2ty_o01.29997173.0.0.4d3b5171SWqFym (87,000 images with 3000 per class)]
- Classes: [A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z, nothing]
- Acquisition: [Manually download and extract the asl_alphabet_train folder into the project directory on Windows.]

## 5. Technical Approach
- Custom CNN with 3 convolutional blocks (Conv → ReLU → MaxPool), followed by dense layers and dropout.
- Model: [From Scratch CNN.]
- Framework: [Tensorflow + OpenCV for video]
- Hardware: [Local Machine with no GPU]

## 6. Expected Challenges & Mitigations
- Challenge: Lacking GPU; Possible Overfitting; Poor Tracking on different users or lighting conditions.
- Solution: Use small image resolution and avoid using heavy architecture; Apply data randomization and incude dropout layer; Encourage stable hand placement and lighting.