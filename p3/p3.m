clc
close all;
clear;
%% Load the image
[file, path] = uigetfile({'*.jpg;*.png;*.jpeg'}, 'Select an image');
imagePath = fullfile(path, file);
img = imread(imagePath);

%% Resize the image 
img = imresize(img, [400, NaN]);
figure, imshow(img);

%% Convert the image to HSV color space (to detect blue more easily)
hsvImage = rgb2hsv(img);
figure, imshow(img);

%% Define thresholds for the blue color in HSV
blueHueMin = 0.58;  % Minimum hue for blue
blueHueMax = 0.75;  % Maximum hue for blue
blueSatMin = 0.4;   % Minimum saturation for blue
blueValMin = 0.2;   % Minimum value for blue (brightness)

%% Create a binary image 
binaryImg = (hsvImage(:,:,1) >= blueHueMin) & (hsvImage(:,:,1) <= blueHueMax) & ...
           (hsvImage(:,:,2) >= blueSatMin) & ...
           (hsvImage(:,:,3) >= blueValMin);

figure, imshow(binaryImg);

%% Label connected components 
[LabeledImg, num] = bwlabel(binaryImg);
figure, imshow(label2rgb(LabeledImg));  % Convert labels to colors for visualization

%% find the largest blue component
maxArea = 0;
blueRegionIdx = 0;
for k = 1:num
    % Find the pixel coordinates of the current component
    [rows, cols] = find(LabeledImg == k);
    
    % Calculate the bounding box (Bounding Box)
    minRow = min(rows);
    maxRow = max(rows);
    minCol = min(cols);
    maxCol = max(cols);
    
    % Compute the width and height of the bounding box
    blueWidth = maxCol - minCol + 1;
    blueHeight = maxRow - minRow + 1;
    
    % Calculate the area of the region (Area)
    area = length(rows);  % Number of pixels corresponds to the area

    % Find the largest region (the blue part of the plate)
    if area > maxArea
        maxArea = area;
        blueRegionIdx = k;
        bluestatus = [minCol, minRow, blueWidth, blueHeight];
    end
end

%% Extract the edges of the plate
% Estimate the right boundary based on the fact that blue part is 1/11 of the plate
blueWidth = bluestatus(3);  % Width of the blue region
plateWidth = blueWidth * 11;  % The whole plate is approximately 11 times the blue region width

% Define the plate bounding box using the top, bottom, left, and calculated right boundary
plateX = round(bluestatus(1));  % The left boundary is the same as the blue region's left boundary
plateY = round(bluestatus(2));  % The top boundary is the same as the blue region's top boundary
plateHeight = round(bluestatus(4));  % Use the same height as the blue region

% Calculate the right boundary by adding the plate width to the left boundary
plateRightX = plateX + round(plateWidth);

% Manually crop the plate using array indexing (rows and columns)
licensePlate = img(plateY:(plateY + plateHeight - 1), plateX:plateRightX, :);  % Crop the region

% Step 10: Display the license plate
figure, imshow(licensePlate);
title('Detected Plate');

% Save the extracted license plate
imwrite(licensePlate, 'detected_plate.png');

disp(' plate detected and saved successfully.');

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% P2 starts
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Part 0: Loading the mapset

% Get a list of all files in the "Map Set" directory
files = dir('persian_dataset');

% Calculate the number of valid files (not '.' and '..')
% The first two elements ('.' and '..') are not image files, so delete them.
len = length(files) - 2;

% Initialize a cell array to store training data.
% The first row will store the images, and the second row will store the corresponding labels (characters).
TRAIN = cell(2, len);

% Loop through each file in the directory, starting from the third file (since the first two are '.' and '..')
for i = 1:len
   % Read the image from the file and store it in the first row of the TRAIN cell array
   TRAIN{1, i} = imread([files(i + 2).folder, '\', files(i + 2).name]);
   
   % Extract the first character of the file name (its the label)
   % and store it in the second row of the TRAIN cell array
   TRAIN{2, i} = files(i + 2).name(1);
end

%% part 0.1 : convert RGB mapset to Binary images and save them again
% Define the threshold value for binarization
threshold = 128;  % Adjust the threshold as needed

% Loop through each image in the TRAIN dataset
for i = 1:size(TRAIN, 2)
    % Extract the RGB image from the training set
    rgbImage = TRAIN{1, i};

    % Convert the RGB image to grayscale using your custom function
    grayImage = mygrayfun(rgbImage);

    % Convert the grayscale image to binary (logical) using your custom function
    binaryImage = mybinaryfun(grayImage, threshold);

    % Replace the RGB image in TRAIN with the binary image
    TRAIN{1, i} = binaryImage;
end

%% part 0.2 : Code to Detect the Bounding Box, Crop, and Resize

% Loop through each image in the TRAIN dataset
for i = 1:size(TRAIN, 2)
    % Extract the binary image from the training set
    binaryImage = TRAIN{1, i};
    
    % Find the rows and columns that contain the value 1
    [rows, cols] = find(binaryImage == 1);

    % If no '1' pixels are found, keep the image as is
    if isempty(rows) || isempty(cols)
        croppedImage = binaryImage;  % No cropping needed, no '1' found
    else
        % Detect the smallest bounding box that contains all the '1' pixels
        minRow = min(rows);
        maxRow = max(rows);
        minCol = min(cols);
        maxCol = max(cols);
        
        % Crop the image to that bounding box
        croppedImage = binaryImage(minRow:maxRow, minCol:maxCol);
    end
    
    % Resize the cropped image back to 150x150
    resizedImage = imresize(croppedImage, [150 150]);

    % Overwrite the original image in the TRAIN cell array with the resized image
    TRAIN{1, i} = resizedImage;
end

% Save the modified TRAIN cell array back to the same TRAININGSET.mat file
save TRAININGSET TRAIN;
%% part 1 : input picture
% build full file path and read the image from the selected file path
img = imread('detected_plate.png');
figure, imshow(img);

%% part 2 : resize
img = imresize(img, [150, 700]);
figure, imshow(img);

%% Convert the color image to grayscale using mygrayfun function
grayImg = mygrayfun(img);
figure, imshow(grayImg);

%% Convert the grayImg to binary image using mybinaryfun function
threshold = 120;  % Selected threshold value 
binaryImage = mybinaryfun(grayImg, threshold);
figure, imshow(binaryImage);

%% Display the cleaned binary image
minSize = 900;  % Minimum size of objects to keep
cleanImage = myremovecom(binaryImage, minSize);
figure, imshow(cleanImage);
title('Cleaned Binary Image (Small Components Removed)');

%% Display the segmented (labeled) image
[labeledImage, numObjects] = mysegmentation(cleanImage);
figure, imshow(label2rgb(labeledImage));  % Convert labels to colors for visualization
title(['Segmented Image (', num2str(numObjects), ' Objects)']);

%% Part 7: Compact Decision-Making Using Correlation (corr2)

% Load the saved training set
load TRAININGSET;
numTemplates = size(TRAIN, 2);

recognizedText = '';  % Store recognized characters
for segmentIndex = 1:numObjects
    [rowIndices, colIndices] = find(labeledImage == segmentIndex);  % Find row and column indices of the current object
   
    % Extract the sub-image (the current segment) from the binary cleanImage.
    % The segment is defined by the bounding box surrounding the object (min/max row and column indices).
    % Then, resize the extracted segment to 150*150 pixels to match the size of the templates in the training set.
    currentSegment = imresize(cleanImage(min(rowIndices):max(rowIndices), min(colIndices):max(colIndices)), [150, 150]);
    
    % Compute correlation scores for all templates
    ro = zeros(1, numTemplates);  % Initialize correlation scores array
    for templateIndex = 1:numTemplates
        ro(templateIndex) = corr2(TRAIN{1, templateIndex}, currentSegment);  % Correlation for each template
    end

 % Create a new figure for each segment with 2 subplots: one for the image, one for the plot
    fig = figure;  % Create a new figure for every segment
    
    % Subplot 1: Show the image of the segment
    subplot(1, 2, 1);  % 1 row, 2 columns, 1st subplot
    imshow(currentSegment);  % Display the segment image
    title(['Segment ', num2str(segmentIndex)]);

    % Subplot 2: Show the correlation scores (ro)
    subplot(1, 2, 2);  % 1 row, 2 columns, 2nd subplot
    bar(ro);  % Bar plot of correlation scores
    title(['Correlation Scores for Segment ', num2str(segmentIndex)]);
    xlabel('Template Label');
    ylabel('Correlation Score');
    
    % Customize the x-axis to display the template labels instead of indices
    xticks(1:numTemplates);  % Set x-ticks at positions corresponding to each template
    xticklabels(TRAIN(2, :));  % Set x-tick labels to the corresponding template labels (characters)

    drawnow;  % Ensure that the figure is rendered before the next iteration
    
    % Find the best match and apply a threshold
    [bestScore, bestMatchIndex] = max(ro);  % Get the best match score and index
    if bestScore > 0.5  % Threshold for accepting a match
        recognizedText = [recognizedText, TRAIN{2, bestMatchIndex}];  % Append detected character
    end
end

%% Part 8: Save the Detected Characters to a File
% Display the result
disp(['Detected pelaak: ', recognizedText]);

% Save the detected characters to a file
fileID = fopen('number_Plate.txt', 'wt');
fprintf(fileID, '%s\n', recognizedText);
fclose(fileID);
winopen('number_Plate.txt');
%% part 3 : build grayscale image
function grayImg = mygrayfun(colorImg)
    % Convert the input color image to grayscale using the weighted sum of color channels
    % grayImg = 0.299 * Red + 0.578 * Green + 0.114 * Blue
    
    % Extract color channels from the input image
    redChannel = colorImg(:,:,1);
    greenChannel = colorImg(:,:,2);
    blueChannel = colorImg(:,:,3);
    
    % Apply the grayscale formula
    grayImg = 0.299 * redChannel + 0.578 * greenChannel + 0.114 * blueChannel;
end

%% part 4: Convert the grayscale image to binary using a threshold
function binaryImage = mybinaryfun(grayImage, threshold)
    binaryImage = grayImage > threshold;  % Convert to binary using the threshold
    binaryImage = ~binaryImage;
end

%% part 5: Remove small connected components (noise) without using bwareaopen
function cleanImage = myremovecom(binaryImage, n_minSize)
    % Step 1: Find all points where the picture is 1 (white pixels in the binary image)
    [row, col] = find(binaryImage == 1);  % Now looking for white pixels (value 1)
    POINTS = [row'; col'];  % Store points as a 2*N matrix (row in first row, col in second row)
    POINTS_NUM = size(POINTS, 2);  % Total number of points

    FINALOBJECT = {};  % To store all connected components
    components_count = 1;  % Counter for connected components
    
    while POINTS_NUM > 0
        % Step 2: Initialize the current object with the first point
        initpoint = POINTS(:, 1);
        POINTS(:, 1) = [];  % Remove the used point
        POINTS_NUM = POINTS_NUM - 1;
        CurrectObject = initpoint;  % Start building the current object
        
        % Step 3: Continue finding new points connected to the current object
        while true
            [POINTS, newPoints] = close_points(initpoint, POINTS);
            newPoints_len = size(newPoints, 2);
            CurrectObject = [CurrectObject newPoints];  % Add new points to the current object
            
            % If no new points are found, stop
            if newPoints_len == 0
                break;
            end
            
            % Explore new points further
            while newPoints_len > 0
                initpoint = newPoints(:, 1);
                newPoints(:, 1) = [];  % Remove the used point
                [POINTS, newPoints2] = close_points(initpoint, POINTS);
                CurrectObject = [CurrectObject newPoints2];  % Add new points to the current object
                newPoints = [newPoints newPoints2];  % Continue with the new points
                newPoints_len = size(newPoints, 2);
            end
        end
        
        % Store the found connected object
        FINALOBJECT{components_count} = CurrectObject;
        components_count = components_count + 1;
        POINTS_NUM = size(POINTS, 2);  % Update the number of remaining points
        
        % If there are more points, initialize a new object
        if POINTS_NUM > 0
            initpoint = POINTS(:, 1);
            CurrectObject = initpoint;
        end
    end
    
    % Step 4: Remove small objects
    cleanImage = zeros(size(binaryImage));  % Initialize a clean image with the same size
    for i = 1:length(FINALOBJECT)
        % Check if the size of the current object is greater than the minimum size
        if size(FINALOBJECT{i}, 2) >= n_minSize
            % Add the object to the clean image
            for j = 1:size(FINALOBJECT{i}, 2)
                cleanImage(FINALOBJECT{i}(1, j), FINALOBJECT{i}(2, j)) = 1;
            end
        end
    end
end

%% part 6: segmenting the picture
function [labeledImage, numObjects] = mysegmentation(binaryImage)
    % Step 1: Initialize the labeled image
    [rows, cols] = size(binaryImage);  % Get image dimensions
    labeledImage = zeros(rows, cols);  % Initialize the labeled image with zeros
    currentLabel = 0;  % Start with label 0

    % Step 2: Find all connected components (black pixels) using myremovecom logic
    [row, col] = find(binaryImage == 1);  % Find all foreground pixels (black, 0)
    POINTS = [row'; col'];  % Store points as a 2xN matrix (row in first row, col in second row)
    POINTS_NUM = size(POINTS, 2);  % Total number of points
    
    while POINTS_NUM > 0
        currentLabel = currentLabel + 1;  % Increment the label for the new object
        initpoint = POINTS(:, 1);  % Start with the first unprocessed point
        POINTS(:, 1) = [];  % Remove the used point
        POINTS_NUM = POINTS_NUM - 1;
        CurrectObject = initpoint;  % Start the current connected component

        % Step 3: Find all connected points using close_points (from myremovecom)
        while true
            [POINTS, newPoints] = close_points(initpoint, POINTS);
            newPoints_len = size(newPoints, 2);
            CurrectObject = [CurrectObject newPoints];  % Add new points to the current object

            if newPoints_len == 0
                break;  % No more connected points found, finish this component
            end

            % Process new points found in this component
            while newPoints_len > 0
                initpoint = newPoints(:, 1);
                newPoints(:, 1) = [];  % Remove the used point
                [POINTS, newPoints2] = close_points(initpoint, POINTS);
                CurrectObject = [CurrectObject newPoints2];  % Add the newly found points
                newPoints = [newPoints newPoints2];  % Continue with the new points
                newPoints_len = size(newPoints, 2);
            end
        end
        
        % Step 4: Label the connected component in labeledImage
        for i = 1:size(CurrectObject, 2)
            labeledImage(CurrectObject(1, i), CurrectObject(2, i)) = currentLabel;  % Assign current label
        end

        POINTS_NUM = size(POINTS, 2);  % Update remaining points
    end
    
    % Step 5: Return the labeled image and the number of objects
    numObjects = currentLabel;  % Total number of labeled objects
end

%% Helper function to find points close to a given point
function [remainingPoints, neighbor_Points] = close_points(initpoint, POINTS)
    % This function returns points in POINTS that are close to
    % the initpoint and remove them from POINTS

    neighbor_Points = [];
    remainingPoints = POINTS;
    for i = size(POINTS, 2):-1:1
        % Check the distance between the initpoint and the current point
        if abs(POINTS(1, i) - initpoint(1)) <= 1 && abs(POINTS(2, i) - initpoint(2)) <= 1
            % Add the point to newPoints and remove it from remainingPoints
            neighbor_Points = [neighbor_Points POINTS(:, i)];
            remainingPoints(:, i) = [];
        end
    end
end



