JHMDBFolder = '..';
subFolder = [JHMDBFolder '/sub_splits/'];
frameFolder = [JHMDBFolder '/Rename_Images'];

splitInd = 1;
for subInd =1:length(splitInd)
    fprintf('SubSplit %d...\n',subInd);
    fileName = strcat('*split_',num2str(subInd),'.txt');
    fileInFolder = strcat(subFolder,fileName);
    subFiles = dir(fileInFolder);
    trainNum = 0;
    testNum = 0;
    
    %Collect Summary information
    dec=0;
    obj=0;

    for i =1:length(subFiles)
       fileName = subFiles(i).name; 
       pos = strfind(fileName,'_test');
       category = fileName(1:pos-1);
       fprintf(' Category %d : %s\n',i,category);
       fid = fopen(strcat(subFolder,fileName));
       seqInCat = 1;

       tline = fgets(fid);
       while ischar(tline)
           seqName = strtok(tline,'.');
           trainTest = str2double( tline(end-1) );
           %fprintf('  Seq %d,train=%d\n',seqInCat,trainTest);

           seqFolder = strcat(JHMDBFolder,'Rename_Images/',category,'/',seqName);
           annoName = strcat(JHMDBFolder,'joint_positions/',category,'/',seqName,'/joint_positions.mat');
           anno = load(annoName);
           
              label = anno.pos_img;

               %Intropolate for Belly %ratio = 0.65 is the best.
               pred_x = squeeze( 0.25 * ( label(1,4,:) + label(1,5,:) + label(1,6,:) + label(1,7,:) ) );
               pred_y = squeeze( 0.5*(label(2,4,:)+label(2,5,:)) + 0.65 * 0.5*(label(2,6,:)+label(2,7,:)-label(2,4,:)-label(2,5,:)) );
               
               for k =1:length(pred_x)
                  bodysize = norm( [label(1,5,k) label(2,5,k)] - [label(1,6,k) label(2,6,k)] );
                 
                    error_dist = norm( [ pred_x(k) pred_y(k) ] - [label(1,2,k) label(2,2,k)] );
                    
                    hit = error_dist <= 0.4*bodysize;
                    
                    if(hit)
                        dec = dec+1;
                    end
                    obj=obj+1;
                    
                    fprintf(' %d',hit);
               end
               fprintf(' |%.4f\n',dec/obj);
           
           seqInCat = seqInCat + 1;
           tline = fgets(fid);
       end
       fclose(fid);
    end
end
