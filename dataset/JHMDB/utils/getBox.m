function [ bbox ] = getBox( mask )
    dim1 = size(mask,1);
    dim2 = size(mask,2);
    
    mask_col = mask(:);
    pos1 = find(mask_col,1,'first');
    pos2 = find(mask_col,1,'last');
    
    mask_row = mask';
    pos3 = find(mask_row,1,'first');
    pos4 = find(mask_row,1,'last');
    
    bbox(1) = floor(pos1/dim1)+1;
    bbox(3) = floor(pos2/dim1)+1;
    bbox(2) = floor(pos3/dim2)+1;
    bbox(4) = floor(pos4/dim2)+1;
end

