% S03_ExplainabilityGradCAM.m
% Visual Interpretability module via Grad-CAM mapping
clc; clear; close all;

disp('Starting S03 Explainability Routine...')

% 0. Create Output Directories
outDir = fullfile('results', 'gradcam');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

% 1. Automatically find the best trained models for ALL experiments (E1-E4)
disp('Scanning training results to find the best models for each experiment...');
csvFiles = dir(fullfile('train', '**', 'experiments_results.csv'));

bestModels = struct(); % Store best model path per experiment

for i = 1:length(csvFiles)
    csvPath = fullfile(csvFiles(i).folder, csvFiles(i).name);
    try
        data = readtable(csvPath);
        for j = 1:height(data)
            if iscell(data.Experiment)
                expId = data.Experiment{j};
                netName = data.Network{j};
            elseif isstring(data.Experiment) || iscategorical(data.Experiment)
                expId = char(data.Experiment(j));
                netName = char(data.Network(j));
            else
                expId = data.Experiment(j);
                netName = data.Network(j);
            end
            
            f1 = data.F1_Score(j);
            epoch = data.Epoch(j);
            modelFile = sprintf('model_%s_%s_ep%d.mat', expId, netName, epoch);
            modelPath = fullfile(csvFiles(i).folder, modelFile);
            
            validExpId = matlab.lang.makeValidName(expId);
            
            % If this experiment ID is not tracked yet, or we found a better F1 score
            if ~isfield(bestModels, validExpId) || f1 > bestModels.(validExpId).F1
                bestModels.(validExpId).F1 = f1;
                bestModels.(validExpId).Path = modelPath;
                bestModels.(validExpId).NetName = netName;
                bestModels.(validExpId).OriginalId = expId;
            end
        end
    catch ME
        warning('Failed to parse %s: %s', csvPath, ME.message);
    end
end

expKeys = fieldnames(bestModels);
if isempty(expKeys)
    error('Could not find any experiment models. Please ensure S02 ran successfully.');
end

fprintf('Found best models for %d experiments: %s\n', length(expKeys), strjoin(expKeys, ', '));

% Load all the trained networks into memory once
fprintf('\nPreloading trained models into memory...\n');
numExps = length(expKeys);
for k = 1:numExps
    validExpId = expKeys{k};
    modelPath = bestModels.(validExpId).Path;
    if exist(modelPath, 'file')
        temp = load(modelPath, 'trainedNet');
        bestModels.(validExpId).Net = temp.trainedNet;
        
        % Extract the correct layer name
        if any(arrayfun(@(l) strcmp(l.Name, 'res5c_relu'), temp.trainedNet.Layers))
            bestModels.(validExpId).LayerName = 'res5c_relu';
        elseif any(arrayfun(@(l) strcmp(l.Name, 'res5b_relu'), temp.trainedNet.Layers))
            bestModels.(validExpId).LayerName = 'res5b_relu';
        else
            reluLayers = arrayfun(@(l) isa(l, 'nnet.cnn.layer.ReLULayer'), temp.trainedNet.Layers);
            layerObjects = temp.trainedNet.Layers(reluLayers);
            bestModels.(validExpId).LayerName = layerObjects(end).Name;
        end
    else
        warning('Model file missing for %s', bestModels.(validExpId).OriginalId);
    end
end

% 2. Open Testing Image via random sampling
disp('Loading image datastore from dataset_original/test...');
imds = imageDatastore(fullfile('dataset_original', 'test'), 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Shuffle and pick 5 images from each class (Non-Tumor, Non-Viable-Tumor, Viable)
imds = splitEachLabel(imds, 5, 'randomized');
numSamples = numel(imds.Files);
sampleFiles = imds.Files;
sampleLabels = imds.Labels;

% Calculate grid size (2x3 for 4 experiments + 1 original)
cols = 3;
rows = ceil((numExps + 1) / cols);

disp('Generating Grad-CAMs for 10 random samples...');

for sampleIdx = 1:numSamples
    imgPath = sampleFiles{sampleIdx};
    actualClass = char(sampleLabels(sampleIdx));
    [~, imgName, ext] = fileparts(imgPath);
    
    imgRaw = imread(imgPath);
    
    % Use an invisible figure to prevent display crashes when rendering many plots
    fig = figure('Name', sprintf('Grad-CAM Sample %d', sampleIdx), 'Position', [100 100 1200 600], 'Visible', 'off');
    
    % Plot Original Image First
    subplot(rows, cols, 1);
    imshow(imgRaw);
    title(sprintf('Original Slide\nClass: %s', actualClass), 'Interpreter', 'none');
    
    % 4. Process each experiment
    for k = 1:numExps
        validExpId = expKeys{k};
        expId = bestModels.(validExpId).OriginalId;
        
        if ~isfield(bestModels.(validExpId), 'Net')
            continue;
        end
        
        trainedNet = bestModels.(validExpId).Net;
        layerName = bestModels.(validExpId).LayerName;
        
        % Get input size and resize
        inputSize = trainedNet.Layers(1).InputSize;
        imgResized = imresize(imgRaw, inputSize(1:2));
        
        % Classify Image Outcome
        [YPred, scores] = classify(trainedNet, imgResized);
        className = char(YPred);
        
        % Deploy Gradient Mapping extraction
        scoreMap = gradCAM(trainedNet, imgResized, className, 'FeatureLayer', layerName);
        
        % Plotting for this experiment
        subplot(rows, cols, k + 1);
        imshow(imgResized);
        hold on;
        hMap = imagesc(scoreMap);
        set(hMap, 'AlphaData', 0.5);
        colormap jet; 
        
        % If predicted incorrectly, make title red
        if strcmp(className, actualClass)
            titleColor = 'k'; % black
        else
            titleColor = 'r'; % red
        end
        
        title(sprintf('%s (%s)\nPred: %s (F1: %.3f)', expId, bestModels.(validExpId).NetName, className, bestModels.(validExpId).F1), 'Color', titleColor, 'Interpreter', 'none');
    end
    
    % Save out
    outName = fullfile(outDir, sprintf('gradcam_sample_%02d_%s.png', sampleIdx, actualClass));
    exportgraphics(fig, outName, 'Resolution', 300);
    close(fig); % Free memory
    
    fprintf('  -> Saved sample %d / %d: %s\n', sampleIdx, numSamples, outName);
end

disp('Execution Finished. Check results/gradcam/ for output images.')