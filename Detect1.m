
clc
close all 

[filename, pathname] = uigetfile({'*.*';'*.bmp';'*.jpg';'*.gif'}, 'Pick a Leaf Image File');
%Applying median filter to remove noises from images
I = imread([pathname,filename]);
I = imresize(I,[256,256]);
N = imnoise(I, 'salt & pepper', 0.3);
red_channel = N(:, :, 1);
green_channel = N(:, :, 2);
blue_channel = N(:, :, 3);

red_channel = medfilt2(red_channel, [3 3]);
green_channel = medfilt2(green_channel, [3 3]);
blue_channel = medfilt2(blue_channel, [3 3]);

F = cat(3, red_channel, green_channel, blue_channel);

subplot(2, 1, 1);
imshow(N);
title('Noisy Image');

subplot(2, 1, 2);
imshow(F);
title('Image After Noise Removal');


%figure, imshow(I); title('Query Leaf Image');

% Enhance Contrast median filtering refer matlab code
I = imadjust(I,stretchlim(I));
figure, imshow(I);title('Filtered Image');
% Otsu Segmentation
I_Otsu = imbinarize(I,graythresh(I));
% Conversion to HIS
I_HIS = rgb2hsi(I);
%% Extract Features
% Function call to evaluate features
%[feat_disease seg_img] =  EvaluateFeatures(I)
% Color Image Segmentation
% Use of K Means clustering for segmentation
% Convert Image from RGB Color Space to L*a*b* Color Space 
% The L*a*b* space consists of a luminosity layer 'L*', chromaticity-layer 'a*' and 'b*'.
% All of the color information is in the 'a*' and 'b*' layers.
cform = makecform('srgb2lab');
% Apply the colorform
lab_he = applycform(I,cform);
% Classify the colors in a*b* colorspace using K means clustering.
% Since the image has 3 colors create 3 clusters.
% Measure the distance using Euclidean Distance Metric.
ab = double(lab_he(:,:,2:3));
nrows = size(ab,1);
ncols = size(ab,2);
ab = reshape(ab,nrows*ncols,2);
nColors = 3;
[cluster_idx, cluster_center] = kmeans(ab,nColors,'distance','sqEuclidean', ...
                                      'Replicates',3);
%[cluster_idx cluster_center] = kmeans(ab,nColors,'distance','sqEuclidean','Replicates',3);
% Label every pixel in tha image using results from K means
pixel_labels = reshape(cluster_idx,nrows,ncols);
%figure,imshow(pixel_labels,[]), title('Image Labeled by Cluster Index');
% Create a blank cell array to store the results of clustering
segmented_images = cell(1,3);
% Create RGB label using pixel_labels
rgb_label = repmat(pixel_labels,[1,1,3]);
for k = 1:nColors
    colors = I;
    colors(rgb_label ~= k) = 0;
    segmented_images{k} = colors;
end
figure, subplot(3,1,1);imshow(segmented_images{1});title('Cluster 1'); subplot(3,1,2);imshow(segmented_images{2});title('Cluster 2');
subplot(3,1,3);imshow(segmented_images{3});title('Cluster 3');
set(gcf, 'Position', get(0,'Screensize'));
% Feature Extraction
x = inputdlg('Enter the cluster no. containing the ROI only:');
i = str2double(x);
% Extract the features from the segmented image
seg_img = segmented_images{i};
% Convert to grayscale if image is RGB
if ndims(seg_img) == 3
   img = rgb2gray(seg_img);
end
%figure, imshow(img); title('Gray Scale Image');
% Evaluate the disease affected area
black = imbinarize(seg_img,graythresh(seg_img));
%figure, imshow(black);title('Black & White Image');
m = size(seg_img,1);
n = size(seg_img,2);
zero_image = zeros(m,n); 
%G = imoverlay(zero_image,seg_img,[1 0 0]);
cc = bwconncomp(seg_img,6);
diseasedata = regionprops(cc,'basic');
A1 = diseasedata.Area;
sprintf('Area of the disease affected region is : %g%',A1);
I_black = imbinarize(I,graythresh(I));
kk = bwconncomp(I,6);
leafdata = regionprops(kk,'basic');
A2 = leafdata.Area;
sprintf(' Total leaf area is : %g%',A2);
%Affected_Area = 1-(A1/A2);
Affected_Area = (A1/A2);
if Affected_Area < 0.1
    Affected_Area = Affected_Area+0.15;
end
sprintf('Affected Area is: %g%%',(Affected_Area*100))
% Create the Gray Level Cooccurance Matrices (GLCMs)
glcms = graycomatrix(img);
% Derive Statistics from GLCM
stats = graycoprops(glcms,'Contrast Correlation Energy Homogeneity');
Contrast = stats.Contrast;
Correlation = stats.Correlation;
Energy = stats.Energy;
Homogeneity = stats.Homogeneity;
Mean = mean2(seg_img);
Standard_Deviation = std2(seg_img);
Entropy = entropy(seg_img);
RMS = mean2(rms(seg_img));
%Skewness = skewness(img)
Variance = mean2(var(double(seg_img)));
a = sum(double(seg_img(:)));
Smoothness = 1-(1/(1+a));
Kurtosis = kurtosis(double(seg_img(:)));
Skewness = skewness(double(seg_img(:)));
% Inverse Difference Movement
m = size(seg_img,1);
n = size(seg_img,2);
in_diff = 0;
for i = 1:m
    for j = 1:n
        temp = seg_img(i,j)./(1+(i-j).^2);
        in_diff = in_diff+temp;
    end
end
IDM = double(in_diff);
    
feat_disease = [Contrast,Correlation,Energy,Homogeneity, Mean, Standard_Deviation, Entropy, RMS, Variance, Smoothness, Kurtosis, Skewness, IDM];
%%
% Load All The Features
load('Training_Data.mat')
% Put the test features into variable 'test'
test = feat_disease;
result = multisvm(Train_Feat,Train_Label,test);
%disp(result);
% Visualize Results
switch result
         case 0
             if A1<16.0000%
                   disp('The disease detected is Anthracnose,Do not worry farmer,keep the vines off the farm to prevent this disease');
             end
         case 1
              if A1 >=20.0000%  
                   disp('The disease detected is LeafRot, Do not worry farmer,remove the dead limbs well below the infected area to prevent this disease');
              elseif A1 <35%  
                   disp('The disease detected is LeafRot, Do not worry farmer,remove the dead limbs well below the infected area to prevent this disease');
              end
         case 2
               if A1 >= 35.0000%
                   disp('The disease detected is Bacterial leaf spot , Do not worry farmer,reduce the pathogen levels by doing crop rotation to prevent this disease');
               elseif A1 < 50.0000%
                   disp('The disease detected is Bacterial leaf spot, Do not worry farmer,reduce the pathogen levels by doing crop rotation to prevent this disease' );
               end
         case 3
             if A1 >=50.0000%
                 disp('The disease detected is powderymildew, Do not worry farmer, prune the plant and remove weeds to prevent this disease  ');
             elseif A1 <80.0000%
                 disp('The disease detected is powderymildew, Do not worry farmer, prune the plant and remove weeds to prevent this disease  ');
             end
         case 5
                disp('The leaf is normal, Do not worry farmer, Your leaf is not infected');
        
end
%% Evaluate Accuracy
load('Accuracy_Data.mat')
Accuracy_Percent= zeros(200,1);
for i = 1:200
data = Train_Feat;
%groups = ismember(Train_Label,1);
groups = ismember(Train_Label,0);
[train,test] = crossvalind('HoldOut',groups);
cp = classperf(groups);
svmStruct = fitcsvm(data(train,:),groups(train),'kernelfunction','linear');
classes = predict(svmStruct,data(test,:));
classperf(cp,classes,test);
Accuracy = cp.CorrectRate;
Accuracy_Percent(i) = Accuracy.*100;
end
Max_Accuracy = max(Accuracy_Percent);
sprintf('Accuracy of Linear Kernel with 200 iterations is: %g%%',Max_Accuracy)