% Step 1: Load the image
[file, path] = uigetfile({'*.jpg;*.png;*.jpeg'}, 'Select an image');
imagePath = fullfile(path, file);
carImage = imread(imagePath);

% Step 2: Resize the image (optional, based on your image size)
carImage = imresize(carImage, [400, NaN]);

% Step 3: Convert the image to HSV color space (easier to identify blue region)
hsvImage = rgb2hsv(carImage);

% Step 4: Define thresholds for the blue color in HSV
% These values may need slight tuning based on the image quality and lighting
blueHueMin = 0.55;  % Minimum hue for blue
blueHueMax = 0.75;  % Maximum hue for blue
blueSatMin = 0.4;   % Minimum saturation for blue
blueValMin = 0.2;   % Minimum value for blue (brightness)

% Step 5: Create a binary mask for the blue region
blueMask = (hsvImage(:,:,1) >= blueHueMin) & (hsvImage(:,:,1) <= blueHueMax) & ...
           (hsvImage(:,:,2) >= blueSatMin) & ...
           (hsvImage(:,:,3) >= blueValMin);

% Step 6: Morphological operations to clean up the mask
se = strel('rectangle', [5, 5]);  % Structuring element for dilation
blueMask = imdilate(blueMask, se);  % Dilate to connect components

% Step 7: Label connected components and find the largest one (the blue region)
[L, num] = bwlabel(blueMask);
stats = regionprops(L, 'BoundingBox', 'Area');

% Find the largest blue region, which corresponds to the blue section of the plate
maxArea = 0;
blueRegionIdx = 0;
for k = 1:num
    if stats(k).Area > maxArea
        maxArea = stats(k).Area;
        blueRegionIdx = k;
    end
end

% Step 8: Extract the bounding box of the blue region
if blueRegionIdx > 0
    blueRegion = stats(blueRegionIdx).BoundingBox;
    
    % Step 9: Assuming the license plate extends to the right of the blue region
    % Extend the bounding box to include the plate
    plateX = blueRegion(1) + blueRegion(3);  % Start at the right edge of the blue region
    plateWidth = size(carImage, 2) - plateX;  % Extend to the right edge of the image
    plateHeight = blueRegion(4);  % Use the same height as the blue region

    % Define the license plate region (right of the blue area)
    plateRegion = [plateX, blueRegion(2), plateWidth, plateHeight];

    % Step 10: Crop and display the license plate
    licensePlate = imcrop(carImage, plateRegion);
    figure, imshow(licensePlate);
    title('Detected License Plate');

    % Save the extracted license plate
    imwrite(licensePlate, 'detected_license_plate.png');
else
    disp('Blue region not detected. No license plate found.');
end
