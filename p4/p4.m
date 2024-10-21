% Step 1: Select and load the video
[videoFile, videoPath] = uigetfile({'*.mp4;*.avi;*.mov'}, 'Select a video file');
videoPath = fullfile(videoPath, videoFile);

% Step 2: Create a VideoReader object
videoObject = VideoReader(videoPath);

% Step 3: Select two different frames (you can modify the frame numbers as needed)
frameNumber1 = 50;  % First frame number (can be modified)
frameNumber2 = 120;  % Second frame number (can be modified)


% Read both frames
frame1 = read(videoObject, frameNumber1);
frame2 = read(videoObject, frameNumber2);

% Step 5: Detect the license plate and find the center for both frames
plateCenter1 = detectPlateAndCenter(frame1);
plateCenter2 = detectPlateAndCenter(frame2);

% Step 6: Display the detected plate centers
if ~isempty(plateCenter1) && ~isempty(plateCenter2)
    fprintf('Center of plate in Frame 1: (%.2f, %.2f)\n', plateCenter1(1), plateCenter1(2));
    fprintf('Center of plate in Frame 2: (%.2f, %.2f)\n', plateCenter2(1), plateCenter2(2));

    % Step 7: Calculate the displacement (movement) of the plate between the two frames
    displacementX = plateCenter2(1) - plateCenter1(1);  % Horizontal movement
    displacementY = plateCenter2(2) - plateCenter1(2);  % Vertical movement

    fprintf('Horizontal movement: %.2f pixels\n', displacementX);
    fprintf('Vertical movement: %.2f pixels\n', displacementY);
    
    frameRate = videoObject.FrameRate;
    frameDifference = abs(frameNumber2 - frameNumber1);  % Difference in frame numbers
    timeDifference = frameDifference / frameRate;  % Time difference in seconds
    speed = sqrt(displacementX ^ 2 + displacementY ^ 2) / timeDifference;
    fprintf('The average speed of the car between frame 50 and 120 is %.2f pixel per second\n', speed);
else
    disp('Failed to detect plate in one or both frames.');
end

% Step 4: Function to detect license plate and return the center of the plate
function plateCenter = detectPlateAndCenter(image)
    % Convert the image to HSV color space for blue detection
    hsvImage = rgb2hsv(image);

    % Define HSV thresholds for blue detection
    blueHueMin = 0.55;  % Minimum hue for blue
    blueHueMax = 0.75;  % Maximum hue for blue
    blueSatMin = 0.4;   % Minimum saturation for blue
    blueValMin = 0.2;   % Minimum value for blue (brightness)

    % Create a binary mask for the blue region
    blueMask = (hsvImage(:,:,1) >= blueHueMin) & (hsvImage(:,:,1) <= blueHueMax) & ...
               (hsvImage(:,:,2) >= blueSatMin) & (hsvImage(:,:,3) >= blueValMin);

    % Morphological operations to clean up the mask
    se = strel('rectangle', [5, 5]);  % Structuring element for dilation
    blueMask = imdilate(blueMask, se);  % Dilate to connect components
    blueMask = imfill(blueMask, 'holes');  % Fill small holes

    % Label connected components and find the largest one (the blue region)
    [L, num] = bwlabel(blueMask);
    stats = regionprops(L, 'BoundingBox', 'Area');

    % Find the largest blue region (likely the blue section of the license plate)
    maxArea = 0;
    blueRegionIdx = 0;
    for k = 1:num
        if stats(k).Area > maxArea
            maxArea = stats(k).Area;
            blueRegionIdx = k;
        end
    end

    % If the blue region is detected, calculate the bounding box of the license plate
    if blueRegionIdx > 0
        blueRegion = stats(blueRegionIdx).BoundingBox;
        blueWidth = blueRegion(3);  % Width of the blue region
        plateWidth = blueWidth * 9;  % The whole plate is approximately 9 times the blue region width

        % Define the plate bounding box using the estimated width and height
        plateX = blueRegion(1);  % The left boundary of the blue region
        plateY = blueRegion(2);  % The top boundary of the blue region
        plateHeight = blueRegion(4);  % Use the same height as the blue region

        % Calculate the right boundary of the license plate
        plateRegion = [plateX, plateY, plateWidth, plateHeight];

        % Extract the center of the bounding box (center of the license plate)
        plateCenterX = plateX + plateWidth / 2;
        plateCenterY = plateY + plateHeight / 2;
        plateCenter = [plateCenterX, plateCenterY];  % Return the center coordinates
        
         % Annotate the image with a red marker at the center of the plate
        figure, imshow(image);  % Display the image
        hold on;
        radius = 5;  % Set the radius of the dot
        rectangle('Position', [plateCenterX - radius, plateCenterY - radius, 2*radius, 2*radius], ...
                  'Curvature', [1, 1], 'EdgeColor', 'r', 'FaceColor', 'r');
        title('Center License Plate');
        hold off;
    else
        plateCenter = [];  % If no plate is detected, return empty
        disp('No plate detected in the frame.');
    end
end


