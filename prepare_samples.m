% Load positive image from PASCAL VOC 2007 dataset
init;

% Get training image file path
im_dir = [VOC07PATH 'JPEGImages/'];
anno_dir = [VOC07PATH 'Annotations/'];
fid = fopen([VOC07PATH 'ImageSets/Main/train.txt']);
train_imgs = textscan(fid, '%s');
train_imgs = train_imgs{1};     % because textscan return 1x1 cell
num_train = length(train_imgs);

% Initialization
counter = 1;
labels = cell(1);
features = cell(1);

% Prepare positive samples, based on RCNN paper only ground truth bounding
% boxes
for ii=1:num_train
    disp(['Images; ' num2str(ii) '/' num2str(num_train)]);
    im_path = [im_dir train_imgs{ii} '.jpg'];
    im = imread(im_path);
    rec = PASreadrecord([anno_dir train_imgs{ii} '.xml']);
    
    num_object = size(rec.objects, 2);
    for jj=1:num_object
        % Get object label and its index
        label = rec.objects(jj).class;
        index = strfind(VOCCLASS, label);
        index = find(not(cellfun('isempty', index)));
        
        box = rec.objects(jj).bndbox;
        patch = im(box.ymin:box.ymax, box.xmin:box.xmax, :);
        
        rep = extract_caffe_feature(patch);
        labels{counter} = index;
        features{counter} = mean(rep, 2);
        counter = counter + 1;
    end
end

save('data/baseline_positive.mat', 'features', 'labels', '-v7.3');

% Prepare negative samples
% As indicated in RCNN paper, regions with less than 0.3 overlap with all
% classes bounding boxes
features = cell(1);
counter = 1;
file_counter = 1;
threshold = 0.3;
for ii=1:num_train
    disp(['Images; ' num2str(ii) '/' num2str(num_train)]);
    im_path = [im_dir train_imgs{ii} '.jpg'];
    im = imread(im_path);
    rec = PASreadrecord([anno_dir train_imgs{ii} '.xml']);
    
    boxes = selective_search(im);
    num_boxes = size(boxes, 1);
    
    for jj=1:num_boxes
        num_object = size(rec.objects, 2);
        reg_box = boxes(jj, :);
        is_background = 1;
        for kk=1:num_object
            obj = rec.objects(kk).bndbox;
            obj_box = [obj.ymin obj.xmin obj.ymax obj.xmax];
            overlap_ratio = compute_overlap(reg_box, obj_box);
            
            if overlap_ratio > threshold
                is_background = 0;
                break;
            end
        end
        
        if is_background == 1
            region = im(reg_box(1):reg_box(3), reg_box(2):reg_box(4), :);
            rep = extract_caffe_feature(region);
            features{counter} = mean(rep, 2);
            counter = counter + 1;
            
            if counter > 100000
                disp(['Save... ' num2str(file_counter)]);
                save(['data/baseline_negative_' num2str(file_counter) '.mat'], 'features', '-v7.3');
                clear features; features = cell(1);
                counter = 1;
                file_counter = file_counter + 1;
            end
        end
    end
end

save(['data/baseline_negative_' num2str(file_counter) '.mat'], 'features', '-v7.3');
