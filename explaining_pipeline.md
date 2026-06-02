# Osteosarcoma Pipeline Explained

This document provides a detailed breakdown of the functions, syntax, and logic used in `Osteosarcoma_Pipeline.m`.

## Overview
`Osteosarcoma_Pipeline.m` is a unified, monolithic MATLAB script that automates the entire machine learning workflow for classifying Osteosarcoma histopathology images. It handles data augmentation, deep transfer learning (using ResNet-50), visual interpretability (Grad-CAM), and comprehensive statistical analysis.

---

## Phase 1: Dataset Balancing
The goal of this phase is to ensure each class has an equal number of images (`targetCount = 1078`) to prevent the model from becoming biased towards a majority class.

### Key Syntax & Functions:
- **`imageDatastore(classSrcDir)`**: Creates a datastore object `imds` that efficiently manages a collection of image files without loading them all into memory at once. It's used here to count existing images (`numel(imds.Files)`).
- **`copyfile(srcBase, balBase)`**: Safely duplicates the original dataset so we can augment images without touching the original files.
- **`readimage(imds, idx)`**: Loads a specific image from the datastore into memory.
- **`randi(numImages)`**: Generates a pseudo-random integer. Used to randomly select an existing image for augmentation.
- **`imwrite(augImg, path)`**: Saves the newly transformed image matrix `augImg` to the disk as a `.png` file.
- **`waitbar(0, 'message')`**: Creates a graphical progress bar to track the augmentation loop.

---

## Phase 2: Model Training
This phase defines the neural network architecture, modifies it for our specific 3-class problem, and trains it across 10 distinct milestones (every 2 epochs up to 20).

### Key Syntax & Functions:
- **`feval('resnet50')`**: Dynamically evaluates and loads the pre-trained ResNet-50 network architecture.
- **`layerGraph(net)`**: Converts the standard network into a modifiable layer graph. This is essential for transfer learning because we need to replace the final layers.
- **`fullyConnectedLayer(...)` & `classificationLayer(...)`**: Creates new final layers tailored to output exactly 3 classes (instead of ResNet's default 1000). The `WeightLearnRateFactor` and `BiasLearnRateFactor` are bumped up to `10` so these new layers learn much faster than the pre-trained frozen layers.
- **`replaceLayer(lgraph, oldName, newLayer)`**: Swaps out the original 1000-class output layers with our newly created 3-class layers.
- **`augmentedImageDatastore(inputSize, imds)`**: Automatically resizes all input images on-the-fly to match the input dimensions required by ResNet-50 (typically 224x224).
- **`trainingOptions('adam', ...)`**: Configures the solver. 
  - `'adam'` is the optimizer.
  - `'MiniBatchSize', 32` dictates how many images are processed at once.
  - `'ExecutionEnvironment', 'gpu'` forces hardware acceleration.
- **`trainNetwork(...)`**: The core function that actually runs the deep learning training loop.
- **`classify(trainedNet, augimdsTest)`**: Runs the newly trained model against the unseen test set to generate predictions (`YPred`).
- **`confusionmat(YTest, YPred)`**: Generates a mathematical NxN matrix comparing the true labels to the predicted labels. This is used to manually calculate Precision, Recall, and F1 Score formulas mathematically.

---

## Phase 3: Explainability Grad-CAM
This phase unpacks the "black box" of the CNN by highlighting the pixels the model looked at when making its decision.

### Key Syntax & Functions:
- **`readtable(csvPath)`**: Reads the generated `experiments_results.csv` to programmatically find which epoch achieved the highest `F1_Score`.
- **`splitEachLabel(imds, 5, 'randomized')`**: Automatically filters the test datastore to grab exactly 5 random images from each of the 3 classes (15 samples total).
- **`arrayfun(@(l) isa(l, 'nnet.cnn.layer.ReLULayer'), bestNet.Layers)`**: An anonymous function (lambda) that maps across all layers in the network to locate the final ReLU (Rectified Linear Unit) activation layer. Grad-CAM requires tapping into these late-stage spatial feature maps.
- **`gradCAM(bestNet, imgResized, className, 'FeatureLayer', layerName)`**: The flagship explainability function. It calculates the gradient of the classification score with respect to the feature map of the chosen layer. It outputs a `scoreMap` (heatmap array).
- **`imagesc(scoreMap)`**: Scales the matrix data to the full range of the current colormap and displays it as an image.
- **`set(hMap, 'AlphaData', 0.5)`**: Makes the heatmap 50% transparent so it can be overlaid on top of the original slide image using `hold on`.

---

## Phase 4: Master Analytics & Evaluation
This phase plots the raw mathematical data generated during Phase 2 into visual artifacts.

### Key Syntax & Functions:
- **`plot(data.Epoch, data.Accuracy, '-o')`**: Generates 2D line plots for the Epoch progressions. `-o` means it will draw a solid line with circle markers at each data point.
- **`confusionchart(...)`**: Different from `confusionmat`. This generates a highly visual, formatted UI chart representing the confusion matrix.
  - `'row-normalized'`: Shows Recall metrics natively.
  - `'column-normalized'`: Shows Precision metrics natively.
- **`bar(metricsData)`**: Generates grouped bar charts. The arrays for Precision, Recall, and F1 are concatenated (`[precClass', recClass', f1Class']`) so they render side-by-side for each specific class.
- **`exportgraphics(fig, filename, 'Resolution', 300)`**: Saves the dynamically generated UI figure to a high-quality (300 DPI) static image file on disk.

---

## Helper Function: `applyCustomAugmentation`
This custom function is called during Phase 1. It mathematically manipulates an image matrix (`img`) to create a "new" synthetic image, effectively expanding the dataset footprint.

### Transformations Applied:
1. **`imrotate(out, angle, 'bilinear', 'crop')`**: Rotates the image. `'crop'` ensures the output matrix size stays identical, cropping the corners that rotate out of bounds.
2. **`fliplr()` / `flipud()`**: Conditionally flips the image matrix left-to-right or up-to-down based on a random `> 0.5` roll.
3. **`imtranslate(out, [tx, ty])`**: Shifts the pixels slightly horizontally and vertically.
4. **`imresize(out, scale)` & `padarray()`**: Zooms in or out. If it zooms out (`scale < 1`), `padarray()` fills the empty border space with `0` (black) to maintain the strict matrix dimensions required for CNN processing.
5. **`rgb2hsv()` & `hsv2rgb()`**: Converts standard Red/Green/Blue color channels into Hue/Saturation/Value. This allows us to easily add mathematical jitter to Brightness (Value) and Color (Hue) without washing out the image, before converting back to RGB.
