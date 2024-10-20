clc
close all;
clear;

%% Part 0: Loading the mapset

% Get a list of all files in the "Map Set" directory
files = dir('Map Set');

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

% Save the TRAIN cell array to a file named "TRAININGSET.mat"
% This will store the images and corresponding labels for future use.
save TRAININGSET TRAIN;

%% part 1 : input picture
[file,path]=uigetfile({'*.jpg;*.png;*.jpeg'},'Select your image');

% build full file path and read the image from the selected file path
img = imread(fullfile(path, file));
figure, imshow(img);

%% part 2 : resize
img = imresize(img, [300, 500]);
figure, imshow(img);

%% Convert the color image to grayscale using mygrayfun function
grayImg = mygrayfun(img);
figure, imshow(grayImg);

%% Convert the grayImg to binary image using mybinaryfun function
threshold = 100;  % Selected threshold value 
binaryImage = mybinaryfun(grayImg, threshold);
figure, imshow(binaryImage);

%% the cleaned binary image
minSize = 300;  % Minimum size of objects to keep
cleanImage = myremovecom(binaryImage, minSize);
figure, imshow(cleanImage);
title('Cleaned Binary Image (Small Components Removed)');

%% the segmented (labeled) image
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
    % Then, resize the extracted segment to 42x24 pixels to match the size of the templates in the training set.
    currentSegment = imresize(cleanImage(min(rowIndices):max(rowIndices), min(colIndices):max(colIndices)), [42, 24]);
    
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
    if bestScore > 0.48  % Threshold for accepting a match
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
function cleanImage = myremovecom(cleanImage, n_minSize)
    % Step 1: Find all points where the picture is 1 (white pixels in the binary image)
    [row, col] = find(cleanImage == 1);  % Get row and column indices of white pixels (value 1)
    POINTS = [row'; col'];  % Store points as a 2*N matrix (row in first row, col in second row)
    
    if isempty(POINTS)
        cleanImage = cleanImage;  % If no points are found, return the original image
        return;
    end

    cleanImage = zeros(size(cleanImage));  % Initialize the clean image (binary)
    
    % Step 2: Process all points and group them into connected components
    while ~isempty(POINTS)
        % Initialize the current object with the first point
        initpoint = POINTS(:, 1);  
        POINTS(:, 1) = [];  % Remove the used point

        % Use a stack to explore the connected component
        points_to_explore = initpoint;
        currentObject = initpoint;

        while ~isempty(points_to_explore)
            % Take the first point from the stack
            current_point = points_to_explore(:, 1);
            points_to_explore(:, 1) = [];  % Remove the used point

            % Find new points connected to the current_point
            [POINTS, newPoints] = close_points(current_point, POINTS);

            % Add the new points to the current object and exploration stack
            currentObject = [currentObject newPoints];
            points_to_explore = [points_to_explore newPoints];
        end
        
        % If the connected component's size is large enough, add it to the clean image
        if size(currentObject, 2) >= n_minSize
            for j = 1:size(currentObject, 2)
                cleanImage(currentObject(1, j), currentObject(2, j)) = 1;
            end
        end
    end
end

%% part 6: segmenting the picture
function [labeledImage, numObjects] = mysegmentation(cleanImage)
    % Step 1: Find all points where the picture is 1 (white pixels in the clean binary image)
    [row, col] = find(cleanImage == 1);  % Get row and column indices of white pixels (value 1)
    POINTS = [row'; col'];  % Store points as a 2*N matrix (row in first row, col in second row)
    
    if isempty(POINTS)
        labeledImage = zeros(size(cleanImage));  % If no points are found, return an empty labeled image
        numObjects = 0;  % No objects detected
        return;
    end

    labeledImage = zeros(size(cleanImage));  % Initialize the labeled image
    currentLabel = 0;  % Start with label 0
    
    % Step 2: Process all points and group them into connected components
    while ~isempty(POINTS)
        % Increment the label for the new object
        currentLabel = currentLabel + 1;

        % Initialize the current object with the first point
        initpoint = POINTS(:, 1);
        POINTS(:, 1) = [];  % Remove the used point

        % Use a stack to explore the connected component
        points_to_explore = initpoint;
        currentObject = initpoint;

        while ~isempty(points_to_explore)
            % Take the first point from the stack
            current_point = points_to_explore(:, 1);
            points_to_explore(:, 1) = [];  % Remove the used point

            % Find new points connected to the current_point
            [POINTS, newPoints] = close_points(current_point, POINTS);

            % Add the new points to the current object and exploration stack
            currentObject = [currentObject newPoints];
            points_to_explore = [points_to_explore newPoints];
        end
        
        % Step 3: Label the connected component in labeledImage
        for j = 1:size(currentObject, 2)
            labeledImage(currentObject(1, j), currentObject(2, j)) = currentLabel;
        end
    end
    
    % Return the labeled image and the number of objects
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



