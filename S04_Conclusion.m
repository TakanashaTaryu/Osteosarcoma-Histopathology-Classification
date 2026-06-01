% S04_Conclusion.m
% Master Analytics, Statistics, and A/B Testing Script
% Generates all quantitative artifacts specified in STATISTICS.md
clc; clear; close all;

disp('=======================================================')
disp('Starting S04 Master Analytics Routine...')
disp('=======================================================')

%% 1. Directory Setup
dirs = {'results/statistics', 'results/training', 'results/comparison', ...
        'results/confusion', 'results/metrics', 'results/performance'};
for i = 1:length(dirs)
    if ~exist(dirs{i}, 'dir'), mkdir(dirs{i}); end
end

%% 2. Load Master Dataset (experiments_results.csv)
disp('Aggregating experiment results...');
csvFiles = dir(fullfile('train', '**', 'experiments_results.csv'));
if isempty(csvFiles)
    error('No experiment results found. Please run S02 first.');
end

masterData = table();
bestModels = struct();

for i = 1:length(csvFiles)
    csvPath = fullfile(csvFiles(i).folder, csvFiles(i).name);
    try
        data = readtable(csvPath);
        masterData = [masterData; data];
        
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
            
            if ~isfield(bestModels, validExpId) || f1 > bestModels.(validExpId).F1
                bestModels.(validExpId).F1 = f1;
                bestModels.(validExpId).Path = modelPath;
                bestModels.(validExpId).NetName = netName;
                bestModels.(validExpId).OriginalId = expId;
                bestModels.(validExpId).Accuracy = data.Accuracy(j);
            end
        end
    catch ME
        warning('Failed to parse %s: %s', csvPath, ME.message);
    end
end

expKeys = fieldnames(bestModels);
disp('Master table generated successfully.');

%% 3. Plotting Training Curves
disp('Generating Epoch Curves (Accuracy & F1)...');
% Ensure Experiment column is string for easier processing
if ~isstring(masterData.Experiment) && ~iscategorical(masterData.Experiment)
    if iscell(masterData.Experiment)
        masterData.Experiment = string(masterData.Experiment);
    else
        masterData.Experiment = string(num2str(masterData.Experiment));
    end
end

uniqueExps = unique(masterData.Experiment);

% Accuracy vs Epoch
figAcc = figure('Name', 'Accuracy vs Epoch', 'Visible', 'off');
hold on; grid on;
for i = 1:length(uniqueExps)
    expRows = masterData(masterData.Experiment == uniqueExps(i), :);
    [~, sortIdx] = sort(expRows.Epoch);
    expRows = expRows(sortIdx, :);
    plot(expRows.Epoch, expRows.Accuracy, '-o', 'LineWidth', 2, 'DisplayName', uniqueExps(i));
end
xlabel('Epoch'); ylabel('Accuracy'); title('Accuracy Progression per Experiment');
legend('Location', 'best');
exportgraphics(figAcc, 'results/comparison/accuracy_vs_epoch.png');
close(figAcc);

% F1 vs Epoch
figF1 = figure('Name', 'F1 Score vs Epoch', 'Visible', 'off');
hold on; grid on;
for i = 1:length(uniqueExps)
    expRows = masterData(masterData.Experiment == uniqueExps(i), :);
    [~, sortIdx] = sort(expRows.Epoch);
    expRows = expRows(sortIdx, :);
    plot(expRows.Epoch, expRows.F1_Score, '-s', 'LineWidth', 2, 'DisplayName', uniqueExps(i));
end
xlabel('Epoch'); ylabel('Macro F1 Score'); title('F1 Score Progression per Experiment');
legend('Location', 'best');
exportgraphics(figF1, 'results/comparison/f1_vs_epoch.png');
close(figF1);

%% 4. Macro F1 Bar Chart
disp('Generating Macro F1 Comparison Chart...');
bestF1Values = zeros(1, length(expKeys));
expLabels = cell(1, length(expKeys));
for k = 1:length(expKeys)
    bestF1Values(k) = bestModels.(expKeys{k}).F1;
    expLabels{k} = bestModels.(expKeys{k}).OriginalId;
end

figMacro = figure('Name', 'Macro F1 Comparison', 'Visible', 'off');
bar(bestF1Values, 'FaceColor', [0.2 0.6 0.5]);
set(gca, 'XTickLabel', expLabels);
ylabel('Max F1 Score'); title('Peak Macro F1 Score Comparison');
ylim([min(bestF1Values)*0.9, min(max(bestF1Values)*1.1, 1)]);
grid on;
exportgraphics(figMacro, 'results/comparison/macro_f1_comparison.png');
close(figMacro);

%% 5. Test Set Evaluation & Confusion Matrices
disp('Loading Test Dataset for Confusion Matrices and A/B Testing...');
testPath = fullfile('dataset_original', 'test');
if ~exist(testPath, 'dir')
    error('dataset_original/test not found! Required for evaluation.');
end
imdsTest = imageDatastore(testPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
YTest = imdsTest.Labels;

% Structure to hold predictions for A/B testing later
predictions = struct();

for k = 1:length(expKeys)
    validExpId = expKeys{k};
    expId = bestModels.(validExpId).OriginalId;
    modelPath = bestModels.(validExpId).Path;
    
    if ~exist(modelPath, 'file')
        warning('Model missing: %s', modelPath);
        continue;
    end
    
    fprintf('  -> Classifying test set using best model from [%s]...\n', expId);
    temp = load(modelPath, 'trainedNet');
    net = temp.trainedNet;
    
    inputSize = net.Layers(1).InputSize;
    augimdsTest = augmentedImageDatastore(inputSize(1:2), imdsTest);
    
    [YPred, ~] = classify(net, augimdsTest);
    predictions.(validExpId) = YPred;
    
    % Generate Confusion Matrix Figure
    figCM = figure('Name', sprintf('Confusion Matrix %s', expId), 'Visible', 'off');
    cm = confusionchart(YTest, YPred, 'Title', sprintf('Confusion Matrix - %s (%s)', expId, bestModels.(validExpId).NetName));
    cm.RowSummary = 'row-normalized';
    cm.ColumnSummary = 'column-normalized';
    exportgraphics(figCM, fullfile('results', 'confusion', sprintf('%s_best_confusion.png', expId)));
    close(figCM);
    
    % Compute per-class metrics
    confMat = confusionmat(YTest, YPred);
    numClasses = size(confMat, 1);
    for c = 1:numClasses
        TP = confMat(c,c);
        FP = sum(confMat(:,c)) - TP;
        FN = sum(confMat(c,:)) - TP;
        bestModels.(validExpId).Precision(c) = TP / max((TP + FP), 1);
        bestModels.(validExpId).Recall(c) = TP / max((TP + FN), 1);
        if (bestModels.(validExpId).Precision(c) + bestModels.(validExpId).Recall(c)) > 0
            bestModels.(validExpId).F1_class(c) = 2 * (bestModels.(validExpId).Precision(c) * bestModels.(validExpId).Recall(c)) / (bestModels.(validExpId).Precision(c) + bestModels.(validExpId).Recall(c));
        else
            bestModels.(validExpId).F1_class(c) = 0;
        end
    end
end

%% 5b. Generate Per-Class Metrics Charts
disp('Generating Per-Class Metrics Charts...');
classNames = categories(YTest);
metricTypes = {'Precision', 'Recall', 'F1_class'};
fileNames = {'precision_per_class.png', 'recall_per_class.png', 'f1_per_class.png'};
titles = {'Precision Per Class', 'Recall Per Class', 'F1 Score Per Class'};

for mIdx = 1:length(metricTypes)
    metricName = metricTypes{mIdx};
    barData = zeros(length(classNames), length(expKeys));
    for k = 1:length(expKeys)
        barData(:, k) = bestModels.(expKeys{k}).(metricName)';
    end
    
    figMetric = figure('Name', titles{mIdx}, 'Visible', 'off');
    bar(barData);
    set(gca, 'XTickLabel', classNames);
    legend(expLabels, 'Location', 'best');
    ylabel(titles{mIdx}); title(titles{mIdx});
    grid on;
    exportgraphics(figMetric, fullfile('results', 'metrics', fileNames{mIdx}));
    close(figMetric);
end

%% 5c. Generate Performance & Training Charts
disp('Generating Performance & Training Charts...');
% 1. Model Size Comparison
modelSizes = zeros(1, length(expKeys));
for k = 1:length(expKeys)
    fileInfo = dir(bestModels.(expKeys{k}).Path);
    modelSizes(k) = fileInfo.bytes / (1024 * 1024); % Convert to MB
end
figSize = figure('Name', 'Model Size Comparison', 'Visible', 'off');
bar(modelSizes, 'FaceColor', [0.4 0.2 0.6]);
set(gca, 'XTickLabel', expLabels);
ylabel('Size (MB)'); title('Model Size Comparison');
grid on;
exportgraphics(figSize, fullfile('results', 'performance', 'model_size_comparison.png'));
close(figSize);

% 2. Individual Training Accuracy & Loss curves (reconstructing from masterData since loss isn't logged, we just plot accuracy per experiment)
for i = 1:length(uniqueExps)
    expStr = char(uniqueExps(i));
    expRows = masterData(masterData.Experiment == uniqueExps(i), :);
    [~, sortIdx] = sort(expRows.Epoch);
    expRows = expRows(sortIdx, :);
    
    figInd = figure('Name', sprintf('Training Curve %s', expStr), 'Visible', 'off');
    plot(expRows.Epoch, expRows.Accuracy, '-o', 'LineWidth', 2);
    xlabel('Epoch'); ylabel('Accuracy'); title(sprintf('Accuracy Curve - %s', expStr));
    grid on;
    exportgraphics(figInd, fullfile('results', 'training', sprintf('accuracy_curve_%s.png', expStr)));
    close(figInd);
end

%% 6. A/B Testing Confidence Intervals (Bootstrapping F1 Scores)
disp('Performing A/B Testing Confidence Intervals on F1 Scores...');
abCsv = fullfile('results', 'statistics', 'ab_testing_results.csv');
fid = fopen(abCsv, 'w');
fprintf(fid, 'Comparison,Model_A,Model_B,Mean_Difference_B_minus_A,CI_Lower,CI_Upper,Statistically_Significant,Winner\n');

% Helper function to calculate Macro F1 from YTest and YPred
calcMacroF1 = @(YT, YP) mean(arrayfun(@(c) ...
    2 * (sum(YT==c & YP==c)/max(sum(YP==c),1)) * (sum(YT==c & YP==c)/max(sum(YT==c),1)) / ...
    max(((sum(YT==c & YP==c)/max(sum(YP==c),1)) + (sum(YT==c & YP==c)/max(sum(YT==c),1))), 1e-10), ...
    unique(YT)));

nBoot = 1000;
alpha = 0.05;
comparisons = {'E1', 'E2'; 'E1', 'E3'; 'E1', 'E4'; 'E2', 'E3'; 'E2', 'E4'; 'E3', 'E4'};

compLabels = cell(1, size(comparisons, 1));
meanDiffs = zeros(1, size(comparisons, 1));
ciLows = zeros(1, size(comparisons, 1));
ciHighs = zeros(1, size(comparisons, 1));

for cIdx = 1:size(comparisons, 1)
    modA = comparisons{cIdx, 1};
    modB = comparisons{cIdx, 2};
    compLabels{cIdx} = sprintf('%s vs %s', modA, modB);
    
    if isfield(predictions, modA) && isfield(predictions, modB)
        yA = predictions.(modA);
        yB = predictions.(modB);
        nTest = length(YTest);
        
        diffBoot = zeros(nBoot, 1);
        for b = 1:nBoot
            idx = randi(nTest, nTest, 1);
            f1A = calcMacroF1(YTest(idx), yA(idx));
            f1B = calcMacroF1(YTest(idx), yB(idx));
            diffBoot(b) = f1B - f1A; % Difference (B - A)
        end
        
        diffBoot = sort(diffBoot);
        ciLows(cIdx) = diffBoot(floor((alpha/2) * nBoot));
        ciHighs(cIdx) = diffBoot(ceil((1 - alpha/2) * nBoot));
        meanDiffs(cIdx) = mean(diffBoot);
        
        isSig = (ciLows(cIdx) > 0 || ciHighs(cIdx) < 0);
        if meanDiffs(cIdx) > 0
            winner = modB;
        else
            winner = modA;
        end
        
        fprintf('Comparison: %s vs %s -> Winner: %s\n', modA, modB, winner);
        fprintf(fid, '%s,%s,%s,%.4f,%.4f,%.4f,%d,%s\n', ...
            compLabels{cIdx}, modA, modB, meanDiffs(cIdx), ciLows(cIdx), ciHighs(cIdx), isSig, winner);
    end
end
fclose(fid);

% Plot the Confidence Intervals
figAB = figure('Name', 'A/B Testing Confidence Intervals', 'Visible', 'off');
hold on;
% Plot zero line for reference
yline(0, 'k--', 'LineWidth', 1.5);
% Error bars where length is distance from mean to CI bounds
negError = meanDiffs - ciLows;
posError = ciHighs - meanDiffs;
errorbar(1:length(compLabels), meanDiffs, negError, posError, 'o', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'b');
set(gca, 'XTick', 1:length(compLabels), 'XTickLabel', compLabels);
ylabel('Mean F1 Difference (Model B - Model A)');
title('95% Confidence Intervals for A/B Testing');
grid on;
exportgraphics(figAB, fullfile('results', 'statistics', 'ab_testing_confidence_intervals.png'));
close(figAB);

disp('=======================================================')
disp('S04 Master Analytics Complete! Check results/ folder.');
disp('=======================================================')
