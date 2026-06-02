clc; clear; close all;

%% Phase 1: Dataset Balancing
disp('--- Phase 1: Dataset Balancing ---');

srcBase = 'dataset_original';
augBase = 'dataset_augmentation';
balBase = 'dataset_balanced';
classes = {'Non-Tumor', 'Non-Viable-Tumor', 'Viable'};
targetCount = 1078; % the target per class

if exist('dataset', 'dir') && ~exist('dataset_original', 'dir')
    movefile('dataset', 'dataset_original');
    disp('Renamed "dataset" to "dataset_original"');
end

if exist(balBase, 'dir')
    disp(['Directory ' balBase ' already exists. Skipping data augmentation and balancing.']);
else
    disp(['Creating ' balBase ' by copying ' srcBase '...']);
    copyfile(srcBase, balBase);

    if ~exist(augBase, 'dir')
        mkdir(augBase);
    end

    trainSrc = fullfile(srcBase, 'train');
    disp('Starting Data Augmentation for Class Balancing...');

    for c = 1:length(classes)
        className = classes{c};
        classSrcDir = fullfile(trainSrc, className);

        imds = imageDatastore(classSrcDir);
        numImages = numel(imds.Files);
        needed = targetCount - numImages;

        if needed > 0
            fprintf('Processing %s | Current: %d | Target: %d | Generating %d images...\n', ...
                className, numImages, targetCount, needed);

            outAugDir = fullfile(augBase, className);
            if ~exist(outAugDir, 'dir'), mkdir(outAugDir); end

            outBalDir = fullfile(balBase, 'train', className);

            wb = waitbar(0, sprintf('Augmenting %s...', className));

            for i = 1:needed
                idx = randi(numImages);
                img = readimage(imds, idx);

                augImg = applyCustomAugmentation(img);
                fname = sprintf('aug_%04d.png', i);

                imwrite(augImg, fullfile(outAugDir, fname));
                imwrite(augImg, fullfile(outBalDir, fname));

                if mod(i, 50) == 0
                    waitbar(i/needed, wb);
                end
            end
            close(wb);
            disp(['>> Successfully balanced ' className ' class!']);
        else
            fprintf('Processing %s | Current: %d. No augmentation needed.\n', className, numImages);
        end
    end
end

%% Phase 2: Model Training
disp('--- Phase 2: Model Training ---');

% Configuration
expId = 'ResNet50_Balanced';
networkModel = 'resnet50';
datasetUsed = 'dataset_balanced';
epochsList = 2:2:20;

outDir = fullfile('train', expId);
if ~exist(outDir, 'dir')
    mkdir(outDir);
end
resultsFile = fullfile(outDir, 'experiments_results.csv');
if ~exist(resultsFile, 'file')
    fid = fopen(resultsFile, 'w');
    fprintf(fid, 'Experiment,Network,Dataset,Epoch,Accuracy,Precision,Recall,F1_Score\n');
    fclose(fid);
end

fprintf('\nInitializing Training: [%s] trained using [%s]\n', networkModel, datasetUsed);

net = feval(networkModel);
fcName = 'fc1000';
classLayerName = 'ClassificationLayer_fc1000'; % resnet50 uses this naming
inputSize = net.Layers(1).InputSize;

trainPath = fullfile(datasetUsed, 'train');
valPath   = fullfile(datasetUsed, 'validate');
testPath  = fullfile(datasetUsed, 'test');

imdsTrain = imageDatastore(trainPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
imdsVal   = imageDatastore(valPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
imdsTest  = imageDatastore(testPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

numClasses = numel(categories(imdsTrain.Labels));

lgraph = layerGraph(net);
newFC = fullyConnectedLayer(numClasses, ...
    'Name', 'new_fc_os', ...
    'WeightLearnRateFactor', 10, ...
    'BiasLearnRateFactor', 10);
newClassLayer = classificationLayer('Name', 'new_classoutput_os');

lgraph = replaceLayer(lgraph, fcName, newFC);
lgraph = replaceLayer(lgraph, classLayerName, newClassLayer);

augimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain);
augimdsVal   = augmentedImageDatastore(inputSize(1:2), imdsVal);
augimdsTest  = augmentedImageDatastore(inputSize(1:2), imdsTest);

currentNet = lgraph;
previousEp = 0;

for epIdx = 1:length(epochsList)
    ep = epochsList(epIdx);
    epochsToTrain = ep - previousEp;
    previousEp = ep;
    fprintf('\n>>> Executing training loop for Epoch %d / %d\n', ep, max(epochsList));

    options = trainingOptions('adam', ...
        'MiniBatchSize', 32, ...
        'MaxEpochs', epochsToTrain, ...
        'InitialLearnRate', 1e-4, ...
        'Shuffle', 'every-epoch', ...
        'ValidationData', augimdsVal, ...
        'ValidationFrequency', max(1, round(numel(imdsTrain.Files)/32)), ...
        'Verbose', true, ...
        'Plots', 'none', ...
        'ExecutionEnvironment', 'gpu');

    trainedNet = trainNetwork(augimdsTrain, currentNet, options);

    YPred = classify(trainedNet, augimdsTest);
    YTest = imdsTest.Labels;

    confMat = confusionmat(YTest, YPred);
    accuracy = sum(diag(confMat)) / sum(confMat(:));

    precisionVec = zeros(numClasses, 1);
    recallVec = zeros(numClasses, 1);
    f1scoreVec = zeros(numClasses, 1);

    for i = 1:numClasses
        TP = confMat(i,i);
        FP = sum(confMat(:,i)) - TP;
        FN = sum(confMat(i,:)) - TP;

        precisionVec(i) = TP / max((TP + FP), 1);
        recallVec(i) = TP / max((TP + FN), 1);

        if (precisionVec(i) + recallVec(i)) > 0
            f1scoreVec(i) = 2 * (precisionVec(i) * recallVec(i)) / (precisionVec(i) + recallVec(i));
        end
    end

    avgPrec = mean(precisionVec);
    avgRec = mean(recallVec);
    avgF1 = mean(f1scoreVec);

    fprintf('[%] Ep %d -> Acc: %.3f | Prec: %.3f | Rec: %.3f | F1: %.3f\n', ...
        ep, accuracy, avgPrec, avgRec, avgF1);

    fid = fopen(resultsFile, 'a');
    fprintf(fid, '%s,%s,%s,%d,%.4f,%.4f,%.4f,%.4f\n', ...
        expId, networkModel, datasetUsed, ep, accuracy, avgPrec, avgRec, avgF1);
    fclose(fid);

    snapName = fullfile(outDir, sprintf('model_%s_%s_ep%d.mat', expId, networkModel, ep));
    save(snapName, 'trainedNet');

    currentNet = layerGraph(trainedNet);
end

%% Phase 3: Explainability Grad-CAM
disp('--- Phase 3: Explainability Grad-CAM ---');
gradDir = fullfile('results', 'gradcam');
if ~exist(gradDir, 'dir')
    mkdir(gradDir);
end

disp('Finding the best model epoch based on F1 Score...');
csvPath = resultsFile;
data = readtable(csvPath);

bestF1 = -1;
bestEpoch = -1;
bestModelPath = '';

for j = 1:height(data)
    f1 = data.F1_Score(j);
    epoch = data.Epoch(j);
    if f1 > bestF1
        bestF1 = f1;
        bestEpoch = epoch;
        bestModelPath = fullfile(outDir, sprintf('model_%s_%s_ep%d.mat', expId, networkModel, epoch));
    end
end

fprintf('Best model found at Epoch %d with F1 Score: %.3f\n', bestEpoch, bestF1);

temp = load(bestModelPath, 'trainedNet');
bestNet = temp.trainedNet;

% Find a suitable ReLU layer for Grad-CAM
if any(arrayfun(@(l) strcmp(l.Name, 'res5c_relu'), bestNet.Layers))
    layerName = 'res5c_relu';
elseif any(arrayfun(@(l) strcmp(l.Name, 'res5b_relu'), bestNet.Layers))
    layerName = 'res5b_relu';
else
    reluLayers = arrayfun(@(l) isa(l, 'nnet.cnn.layer.ReLULayer'), bestNet.Layers);
    layerObjects = bestNet.Layers(reluLayers);
    layerName = layerObjects(end).Name;
end

disp('Loading image datastore from dataset_original/test...');
imdsGC = imageDatastore(fullfile('dataset_original', 'test'), 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
imdsGC = splitEachLabel(imdsGC, 5, 'randomized');
numSamples = numel(imdsGC.Files);

disp('Generating Grad-CAM overlays for 15 random samples...');
for sampleIdx = 1:numSamples
    imgPath = imdsGC.Files{sampleIdx};
    actualClass = char(imdsGC.Labels(sampleIdx));

    imgRaw = imread(imgPath);
    fig = figure('Name', sprintf('Grad-CAM Sample %d', sampleIdx), 'Position', [100 100 800 400], 'Visible', 'off');

    subplot(1, 2, 1);
    imshow(imgRaw);
    title(sprintf('Original Slide\nClass: %s', actualClass), 'Interpreter', 'none');

    imgResized = imresize(imgRaw, inputSize(1:2));
    [YPred, ~] = classify(bestNet, imgResized);
    className = char(YPred);

    scoreMap = gradCAM(bestNet, imgResized, className, 'FeatureLayer', layerName);

    subplot(1, 2, 2);
    imshow(imgResized);
    hold on;
    hMap = imagesc(scoreMap);
    set(hMap, 'AlphaData', 0.5);
    colormap jet;

    titleColor = 'k';
    if ~strcmp(className, actualClass)
        titleColor = 'r';
    end
    title(sprintf('Pred: %s\n(Epoch %d)', className, bestEpoch), 'Color', titleColor, 'Interpreter', 'none');

    outName = fullfile(gradDir, sprintf('gradcam_sample_%02d_%s.png', sampleIdx, actualClass));
    exportgraphics(fig, outName, 'Resolution', 300);
    close(fig);
end
fprintf('Grad-CAM overlays saved to %s\n', gradDir);

%% Phase 4: Master Analytics & Evaluation
disp('--- Phase 4: Analytics & Evaluation ---');
dirs = {'results/statistics', 'results/training', 'results/confusion', 'results/metrics', 'results/performance'};
for i = 1:length(dirs)
    if ~exist(dirs{i}, 'dir'), mkdir(dirs{i}); end
end

disp('Generating Epoch-Based Progression Curves...');
% Ensure the data is sorted by Epoch
[~, sortIdx] = sort(data.Epoch);
data = data(sortIdx, :);

% Accuracy vs Epoch
figAcc = figure('Name', 'Accuracy vs Epoch', 'Visible', 'off');
plot(data.Epoch, data.Accuracy, '-o', 'LineWidth', 2, 'DisplayName', 'Accuracy');
xlabel('Epoch'); ylabel('Accuracy'); title('Accuracy Progression across Epochs');
grid on; legend('Location', 'best');
exportgraphics(figAcc, 'results/training/accuracy_vs_epoch.png');
close(figAcc);

% F1 vs Epoch
figF1 = figure('Name', 'F1 Score vs Epoch', 'Visible', 'off');
plot(data.Epoch, data.F1_Score, '-s', 'LineWidth', 2, 'DisplayName', 'Macro F1');
xlabel('Epoch'); ylabel('Macro F1 Score'); title('F1 Score Progression across Epochs');
grid on; legend('Location', 'best');
exportgraphics(figF1, 'results/training/f1_vs_epoch.png');
close(figF1);

% Evaluate Best Epoch Model on Test Set
disp('Evaluating best model on test set for Confusion Matrix & Per-Class Metrics...');
augimdsTest = augmentedImageDatastore(inputSize(1:2), imdsTest);
[YPredBest, ~] = classify(bestNet, augimdsTest);
YTestBest = imdsTest.Labels;

% Confusion Matrix
figCM = figure('Name', 'Best Model Confusion Matrix', 'Visible', 'off');
cm = confusionchart(YTestBest, YPredBest, 'Title', sprintf('Confusion Matrix - %s (Epoch %d)', networkModel, bestEpoch));
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';
exportgraphics(figCM, fullfile('results', 'confusion', 'best_model_confusion.png'));
close(figCM);

% Per-Class Metrics
confMat = confusionmat(YTestBest, YPredBest);
numClasses = size(confMat, 1);
precClass = zeros(1, numClasses);
recClass = zeros(1, numClasses);
f1Class = zeros(1, numClasses);
classNames = categories(YTestBest);

for c = 1:numClasses
    TP = confMat(c,c);
    FP = sum(confMat(:,c)) - TP;
    FN = sum(confMat(c,:)) - TP;
    precClass(c) = TP / max((TP + FP), 1);
    recClass(c) = TP / max((TP + FN), 1);
    if (precClass(c) + recClass(c)) > 0
        f1Class(c) = 2 * (precClass(c) * recClass(c)) / (precClass(c) + recClass(c));
    end
end

% Plotting Per-Class Metrics
figMetric = figure('Name', 'Per-Class Metrics', 'Visible', 'off');
metricsData = [precClass', recClass', f1Class'];
b = bar(metricsData);
b(1).FaceColor = [0.2 0.6 0.5];
b(2).FaceColor = [0.8 0.4 0.2];
b(3).FaceColor = [0.2 0.4 0.8];
set(gca, 'XTickLabel', classNames);
legend({'Precision', 'Recall', 'F1 Score'}, 'Location', 'best');
ylabel('Score'); title(sprintf('Per-Class Metrics (Epoch %d)', bestEpoch));
grid on;
exportgraphics(figMetric, fullfile('results', 'metrics', 'per_class_metrics.png'));
close(figMetric);

disp('done')

%% Helper Functions
function out = applyCustomAugmentation(img)
if size(img, 3) == 1, img = repmat(img, 1, 1, 3); end

out = im2double(img);

angles = [-30, -20, -10, 10, 20, 30];
angle = angles(randi(length(angles)));
out = imrotate(out, angle, 'bilinear', 'crop');

if rand > 0.5, out = fliplr(out); end
if rand > 0.5, out = flipud(out); end

tx = round((rand * 10) - 5);
ty = round((rand * 10) - 5);
out = imtranslate(out, [tx, ty]);

scale = 0.9 + (rand * 0.2);
outResized = imresize(out, scale);

[h, w, ~] = size(out);
[hr, wr, ~] = size(outResized);
if scale > 1
    r1 = floor((hr - h)/2) + 1;  c1 = floor((wr - w)/2) + 1;
    out = outResized(r1:r1+h-1, c1:c1+w-1, :);
else
    padR = floor((h - hr)/2);    padC = floor((w - wr)/2);
    out = padarray(outResized, [padR, padC], 0, 'both');
    out = imresize(out, [h w]);
end

hsv = rgb2hsv(out);
hsv(:,:,3) = hsv(:,:,3) + ((rand * 0.2) - 0.1);
hsv(:,:,3) = hsv(:,:,3) * (0.8 + rand * 0.4);
hsv(:,:,1) = hsv(:,:,1) + ((rand * 0.1) - 0.05);
hsv = min(max(hsv, 0), 1);

out = im2uint8(hsv2rgb(hsv));
end
