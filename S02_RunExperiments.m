% S02_RunExperiments.m
% Implements the experimental execution plan for Osteosarcoma model training
clc; clear; close all;

% Configuration according to PLAN.md
experiments = struct( ...
    'id', {'E1', 'E2', 'E3', 'E4'}, ...
    'network', {'resnet18', 'resnet18', 'resnet50', 'resnet50'}, ...
    'dataset', {'dataset_original', 'dataset_balanced', 'dataset_original', 'dataset_balanced'} ...
    );

epochsList = 4:4:20;


% Begin Experiment Loop
for expIdx = 1:length(experiments)
    currExp = experiments(expIdx);

    % Create output directory
    outDir = fullfile('train', currExp.id);
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end
    resultsFile = fullfile(outDir, 'experiments_results.csv');
    if ~exist(resultsFile, 'file')
        fid = fopen(resultsFile, 'w');
        fprintf(fid, 'Experiment,Network,Dataset,Epoch,Accuracy,Precision,Recall,F1_Score\n');
        fclose(fid);
    end

    fprintf('\n=======================================================\n');
    fprintf('Initializing %s: [%s] trained using [%s]\n', currExp.id, currExp.network, currExp.dataset);
    fprintf('=======================================================\n');

    % Prepare the network model architecture dynamically
    net = feval(currExp.network);
    if strcmp(currExp.network, 'resnet18')
        fcName = 'fc1000';
        classLayerName = 'ClassificationLayer_predictions';
    else
        fcName = 'fc1000';
        classLayerName = 'ClassificationLayer_fc1000';
    end
    inputSize = net.Layers(1).InputSize;

    % Access matching datastores
    trainPath = fullfile(currExp.dataset, 'train');
    valPath   = fullfile(currExp.dataset, 'validate');
    testPath  = fullfile(currExp.dataset, 'test');

    imdsTrain = imageDatastore(trainPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
    imdsVal   = imageDatastore(valPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
    imdsTest  = imageDatastore(testPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

    numClasses = numel(categories(imdsTrain.Labels));

    % Freeze and Modify the Final Classification Layers for Transfer Learning
    lgraph = layerGraph(net);
    newFC = fullyConnectedLayer(numClasses, ...
        'Name', 'new_fc_os', ...
        'WeightLearnRateFactor', 10, ...
        'BiasLearnRateFactor', 10);
    newClassLayer = classificationLayer('Name', 'new_classoutput_os');

    lgraph = replaceLayer(lgraph, fcName, newFC);
    lgraph = replaceLayer(lgraph, classLayerName, newClassLayer);

    % Incorporate generic native resizer for standard test passing sizes
    augimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain);
    augimdsVal   = augmentedImageDatastore(inputSize(1:2), imdsVal);
    augimdsTest  = augmentedImageDatastore(inputSize(1:2), imdsTest);

    % Initialize the network for the current experiment
    currentNet = lgraph;
    previousEp = 0;

    % Train across epochs, one epoch at a time, to evaluate and log per epoch
    for epIdx = 1:length(epochsList)
        ep = epochsList(epIdx);
        epochsToTrain = ep - previousEp;
        previousEp = ep;
        fprintf('\n>>> Executing %s loop for Epoch %d / %d\n', currExp.id, ep, max(epochsList));

        options = trainingOptions('adam', ...
            'MiniBatchSize', 32, ...
            'MaxEpochs', epochsToTrain, ... % Train for delta epochs
            'InitialLearnRate', 1e-4, ...
            'Shuffle', 'every-epoch', ...
            'ValidationData', augimdsVal, ...
            'ValidationFrequency', max(1, round(numel(imdsTrain.Files)/32)), ...
            'Verbose', true, ...
            'Plots', 'none', ...
            'ExecutionEnvironment', 'gpu'); % Force multi-threaded GPU training

        % Kickoff deep transfer learning sequence for this epoch
        trainedNet = trainNetwork(augimdsTrain, currentNet, options);

        % Predict and Extract Validation Results
        YPred = classify(trainedNet, augimdsTest);
        YTest = imdsTest.Labels;

        % Generate Mathematical Conf Matrices
        confMat = confusionmat(YTest, YPred);
        accuracy = sum(diag(confMat)) / sum(confMat(:));

        precisionVec = zeros(numClasses, 1);
        recallVec = zeros(numClasses, 1);
        f1scoreVec = zeros(numClasses, 1);

        for i = 1:numClasses
            TP = confMat(i,i);
            FP = sum(confMat(:,i)) - TP;
            FN = sum(confMat(i,:)) - TP;

            precisionVec(i) = TP / max((TP + FP), 1); % Denominator protections
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

        % Register into CSV Logs
        fid = fopen(resultsFile, 'a');
        fprintf(fid, '%s,%s,%s,%d,%.4f,%.4f,%.4f,%.4f\n', ...
            currExp.id, currExp.network, currExp.dataset, ep, accuracy, avgPrec, avgRec, avgF1);
        fclose(fid);

        % Ensure state snapshots are permanently collected
        snapName = fullfile(outDir, sprintf('model_%s_%s_ep%d.mat', currExp.id, currExp.network, ep));
        save(snapName, 'trainedNet');

        % Set the trained network as the starting point for the next epoch
        currentNet = layerGraph(trainedNet);
    end
end

disp('=======================================================')
disp('Analysis Plan Output Sequence Complete!');
disp('Check experiments_results.csv for a breakdown table');