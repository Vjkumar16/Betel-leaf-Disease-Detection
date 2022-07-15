% Features Extraction For Training Data Of SVM algorithm
clc;
clear all;
close all;
%Features of Bactrial leaf spot
for i=1:10
    disp(['Processing frame no.',num2str(i)]);
    img=imread(['Anthracnose',num2str(i),'.jpg']);
    img = imresize(img,[256,256]);
    imshow(img);title('Leaf Image');
    [feat_disease, seg_img] =  EvaluateFeatures(img);
    Anthracnose(i,:) = feat_disease;
    save Anthracnose;
    close all
end


% Features of Bacterialleafspot
for i=1:7
    disp(['Processing frame no.',num2str(i)]);
    img=imread(['Bacterialleafspot',num2str(i),'.jpg']);
    img = imresize(img,[256,256]);
    imshow(img);title('Leaf Image');
    [feat_disease, seg_img] =  EvaluateFeatures(img);
    Bacterialleafspot(i,:) = feat_disease;
    save Bacterialleafspot_Feat.mat;
    close all
end

% Features of powderymildew
for i=1:7
    disp(['Processing frame no.',num2str(i)]);
    img=imread(['powderymildew',num2str(i),'.jpg']);
    img = imresize(img,[256,256]);
    imshow(img);title('Leaf Image');
    [feat_disease, seg_img] =  EvaluateFeatures(img);
    powderymildew(i,:) = feat_disease;
    save powderymildew;
    close all
end

% Features of LeafRot
for i=1:9
    disp(['Processing frame no.',num2str(i)]);
    img=imread(['LeafRot',num2str(i),'.jpg']);
    img = imresize(img,[256,256]);
    imshow(img);title('Leaf Image');
    [feat_disease, seg_img] =  EvaluateFeatures(img);
    LeafRot(i,:) = feat_disease;
    save LeafRot;
    close all
end

% Features of HealthyBetelLeaf
for i=1:15
    disp(['Processing frame no.',num2str(i)]);
    img=imread(['HealthyBetelLeaf',num2str(i),'.jpg']);
    img = imresize(img,[256,256]);
    imshow(img);title('Leaf Image');
    [feat_disease, seg_img] =  EvaluateFeatures(img);
    HealthyBetelLeaf(i,:) = feat_disease;
    save HealthyBetelLeaf;
    close all
end