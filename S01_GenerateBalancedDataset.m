% S01_GenerateBalancedDataset.m
% Implements Phase 1 of PLAN.md: Balancing the dataset via offline targeted data augmentation.
clc; clear; close all;

% 1. Rename 'dataset' to 'dataset_original' if necessary to completely match PLAN.md
if exist('dataset', 'dir') && ~exist('dataset_original', 'dir')
    movefile('dataset', 'dataset_original');
    disp('Renamed "dataset" to "dataset_original" to match PLAN.md');
end

% Definitions
srcBase = 'dataset_original';
augBase = 'dataset_augmentation';
balBase = 'dataset_balanced';
classes = {'Non-Tumor', 'Non-Viable-Tumor', 'Viable'};
targetCount = 1078; % the target per class

% 2. Prepare targeting balanced directory by copying original
if ~exist(balBase, 'dir')
    disp(['Creating ' balBase ' by copying ' srcBase '...']);
    copyfile(srcBase, balBase);
end

% 3. Initialize separate augmentation directory for tracking original vs generated
if ~exist(augBase, 'dir')
    mkdir(augBase);
end

trainSrc = fullfile(srcBase, 'train');

disp('Starting Data Augmentation for Class Balancing...');
disp('-----------------------------------------------');

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
        
        % Using waitbar for simple progress tracking
        wb = waitbar(0, sprintf('Augmenting %s...', className));
        
        for i = 1:needed
            idx = randi(numImages); % Randomly select an image to augment
            img = readimage(imds, idx);
            
            % Generate transformation
            augImg = applyCustomAugmentation(img);
            
            % Format names as stated in PLAN.md (e.g. aug_0001.png)
            fname = sprintf('aug_%04d.png', i);
            
            % Save to dataset_augmentation AND dataset_balanced
            imwrite(augImg, fullfile(outAugDir, fname));
            imwrite(augImg, fullfile(outBalDir, fname));
            
            if mod(i, 50) == 0
                waitbar(i/needed, wb);
            end
        end
        close(wb);
        disp(['>> Successfully balanced ' className ' class!']);
    else
        fprintf('Processing %s | Current: %d. No augmentation needed (Limit Reached).\n', ...
            className, numImages);
    end
end
disp('-----------------------------------------------');
disp('Dataset architecture successfully matches PLAN.md.');


% Local Custom Augmentation Function implementing PLAN.md rules
function out = applyCustomAugmentation(img)
    % Validate image is RGB
    if size(img, 3) == 1, img = repmat(img, 1, 1, 3); end
    
    out = im2double(img);
    
    % 1. Rotation (±10°, ±20°, ±30°)
    angles = [-30, -20, -10, 10, 20, 30];
    angle = angles(randi(length(angles)));
    out = imrotate(out, angle, 'bilinear', 'crop');
    
    % 2. Flipping Options
    if rand > 0.5, out = fliplr(out); end % Horizontal Flip
    if rand > 0.5, out = flipud(out); end % Vertical Flip
    
    % 3. Translation (±5 pixels)
    tx = round((rand * 10) - 5);
    ty = round((rand * 10) - 5);
    out = imtranslate(out, [tx, ty]);
    
    % 4. Zoom (0.9x - 1.1x)
    scale = 0.9 + (rand * 0.2);
    outResized = imresize(out, scale);
    
    % Restoring resolution back to original to avoid mismatch crashes during CNN Training
    [h, w, ~] = size(out);
    [hr, wr, ~] = size(outResized);
    if scale > 1
        r1 = floor((hr - h)/2) + 1;  c1 = floor((wr - w)/2) + 1;
        out = outResized(r1:r1+h-1, c1:c1+w-1, :);
    else
        padR = floor((h - hr)/2);    padC = floor((w - wr)/2);
        out = padarray(outResized, [padR, padC], 0, 'both');
        out = imresize(out, [h w]); % resize to exact match if off by 1 array size
    end
    
    % 5. Contrast, Brightness and Color Jitter via HSV conversion
    hsv = rgb2hsv(out);
    hsv(:,:,3) = hsv(:,:,3) + ((rand * 0.2) - 0.1);    % Brightness Variation
    hsv(:,:,3) = hsv(:,:,3) * (0.8 + rand * 0.4);      % Contrast Variation
    hsv(:,:,1) = hsv(:,:,1) + ((rand * 0.1) - 0.05);   % Color Hue Jitter Variation
    hsv = min(max(hsv, 0), 1);                         % Restrict to valid clipping 
    
    out = im2uint8(hsv2rgb(hsv));
end