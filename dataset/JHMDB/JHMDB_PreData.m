%%This is the code for preparing subJHMDB Train && Test sets.%%

addpath('./utils');
JHMDBFolder = '.';
subFolder = [JHMDBFolder '/sub_splits/'];
frameFolder = [JHMDBFolder '/Rename_Images'];
maskFolder = [JHMDBFolder '/puppet_mask/'];

splitInd = [1 2 3];
for subInd =1:length(splitInd)
    fprintf('SubSplit %d...\n',subInd);
    fileName = strcat('*split_',num2str(subInd),'.txt');
    fileInFolder = strcat(subFolder,fileName);
    subFiles = dir(fileInFolder);
    trainNum = 0;
    testNum = 0;

    for i =1:length(subFiles)
       fileName = subFiles(i).name; 
       pos = strfind(fileName,'_test');
       category = fileName(1:pos-1);
       fprintf(' Category %d : %s\n',i,category);
       fid = fopen(strcat(subFolder,fileName));

       tline = fgets(fid);
       while ischar(tline)
           seqName = strtok(tline,'.');
           trainTest = str2double( tline(end-1) );

           seqFolder = strcat(JHMDBFolder,'/Rename_Images/',category,'/',seqName);
           annoName = strcat(JHMDBFolder,'/joint_positions/',category,'/',seqName,'/joint_positions.mat');
           anno = load(annoName);
                    
           nframes = length(anno.scale);
           
           imgSummary = dir( strcat(seqFolder,'/*.png'));
           if(nframes ~= length(imgSummary))
               error('Label length and frame length do not match...\n');
           end    
           for frame = 1:nframes
              imgName = strcat(seqFolder,'/',imgSummary(frame).name);
              try
                img = imread(imgName);
              catch
                error('Error in image reading..\n');
              end
              image(:,:,:,frame) = img;
           end
           
           %Interpolate the bounding box from mask label of JHMDB dataset
           maskName = strcat(maskFolder,category,'/',seqName,'/puppet_mask.mat');
           try
               maskLabel = load(maskName);
               mask = maskLabel.part_mask;
               clear maskLabel;
               if(nframes ~= size(mask,3))
                   error('mask length and frame length do not match...\n');
               end
           catch
               error('Can not find the bounding box for sequence %s...\n',seqName);
           end
                    %figure(1);hold on;subplot(1,2,1);imshow(image(:,:,:,10));subplot(1,2,2);imshow(mask(:,:,10));
           bbox = zeros([nframes 4]);
           for frame =1:nframes
               bbox(frame,:) = getBox(mask(:,:,frame));
           end    
           
           if(trainTest == 1)
               trainNum = trainNum + 1;
               train.sequences{trainNum,1}.frameAdd = seqFolder;
               train.sequences{trainNum,1}.labelAdd = annoName;
               train.sequences{trainNum,1}.train = trainTest;
               train.sequences{trainNum,1}.viewpoint = anno.viewpoint;
               train.sequences{trainNum,1}.nframes = nframes;
               train.sequences{trainNum,1}.pos_img = anno.pos_img;
               train.sequences{trainNum,1}.pos_world = anno.pos_world;
               train.sequences{trainNum,1}.image = image;
               train.sequences{trainNum,1}.scale = anno.scale;
               train.sequences{trainNum,1}.bbox = bbox;
           elseif(trainTest == 2)
               testNum = testNum + 1;
               test.sequences{testNum,1}.frameAdd = seqFolder;
               test.sequences{testNum,1}.labelAdd = annoName;
               test.sequences{testNum,1}.train = trainTest;
               test.sequences{testNum,1}.viewpoint = anno.viewpoint;
               test.sequences{testNum,1}.nframes = nframes;
               test.sequences{testNum,1}.pos_img = anno.pos_img;
               test.sequences{testNum,1}.pos_world = anno.pos_world;
               test.sequences{testNum,1}.image = image;
               test.sequences{testNum,1}.scale = anno.scale;
               test.sequences{testNum,1}.bbox = bbox;
           else
               error('Please check the file %s ...\n',seqFolder);
           end
           tline = fgets(fid);
       end
       fclose(fid);
    end
    fprintf('Writing data for subsplit %d...\n',subInd);
    dataFolder = strcat(JHMDBFolder,'/Sub',num2str(subInd));
    mkdir(dataFolder);
    fileTrain = sprintf(strcat(dataFolder,'/train.mat'));
    save(fileTrain, 'train','-v7.3');
    fileTest = sprintf(strcat(dataFolder,'/test.mat'));
    save(fileTest, 'test','-v7.3');
    fprintf('Write Finish...\n');
    clear train test;
end
