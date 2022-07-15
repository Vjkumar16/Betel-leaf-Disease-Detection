% Training Part
%Features of Anthracnose
for i=1:10
    disp(['Processing frame no.',num2str(i)]);
    img=imread(['Anthracnose',num2str(i),'.jpg']);
    img = imresize(img,[256,256]);
    img = imadjust(img,stretchlim(img));
    %imshow(img);title('Anthracnose');
    %[feat_disease seg_img] =  EvaluateFeatures(img);
    %Anthracnose(i,:) = feat_disease;
    save Anthracnose_Feat;
    close all
end

% Features of Bacterialleafspot
for i=1:7
    disp(['Processing frame no.',num2str(i)]);
    img=imread(['Bacterialleafspot',num2str(i),'.jpg']);
    img = imresize(img,[256,256]);
    img = imadjust(img,stretchlim(img));
    %imshow(img);title('Bacterialleafspot');
    %[feat_disease seg_img] =  EvaluateFeatures(img);
    %Bacterialleafspot(i,:) = feat_disease;
    save Bacterialleafspot_Feat;
    close all
end

% Features of powderymildew
for i=1:8
    disp(['Processing frame no.',num2str(i)]);
    img=imread(['powderymildew',num2str(i),'.jpg']);
    img = imresize(img,[256,256]);
    img = imadjust(img,stretchlim(img));
    %imshow(img);title('powderymildew');
    %[feat_disease seg_img] =  EvaluateFeatures(img);
    %powderymildew(i,:) = feat_disease;
    save powderymildew_Feat;
    close all
end

% Featurs of LeafRot
for i=1:9
    disp(['Processing frame no.',num2str(i)]);
    img=imread(['LeafRot',num2str(i),'.jpg']);
    img = imresize(img,[256,256]);
    img = imadjust(img,stretchlim(img));
    %imshow(img);title('LeafRot');
    %[feat_disease seg_img] =  EvaluateFeatures(img);
    %LeafRot(i,:) = feat_disease;
    save LeafRot_Feat;
    close all
end

% Features of Healthy Image
for i=1:15
    disp(['Processing frame no.',num2str(i)]);
    img=imread(['HealthyBetelLeaf',num2str(i),'.jpg']);
    img = imresize(img,[256,256]);
    img = imadjust(img,stretchlim(img));
    %imshow(img);title('Healthy Leaf Image');
    %[feat_disease seg_img] =  EvaluateFeatures(img);
    %Healthy_Feat(i,:) = feat_disease;
    save HealthyBetelLeaf_Feat;
    close all
end

% Accuracy Evaluation Dataset Preparation
close all
clear all
clc
load('Anthracnose_Feat.mat')
load('Bacterialleafspot_Feat.mat')
load('powderymildew_Feat.mat')
load('LeafRot_Feat.mat')
load('HealthyBetelLeaf_Feat.mat')

%Train_Feat = [Anthracnose_Feat;Bacterialleafspot_Feat;powderymildew_Feat;LeafRot_Feat;HealthyBetelLeaf_Feat];
Train_Label = [ zeros(100,1); ones(25,1) ];
save Accuracy_Data

