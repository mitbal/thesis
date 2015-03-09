% Print detection to file
init;

% Get test image path
testing_data = 'test';
im_dir = [VOC07PATH 'JPEGImages/'];
fid = fopen([VOC07PATH 'ImageSets/Main/' testing_data '.txt']);
test_imgs = textscan(fid, '%s');
test_imgs = test_imgs{1};

% Load trained SVM models
net_model = 'caffenet';
training_data = 'trainval';
load(['svm_models/' net_model '/' training_data '.mat']);

num_files = 496;
img_per_file = 10;
for ii=1:1
    disp(['Class ' VOCCLASS{ii}]);
    fid = fopen(['results/' net_model '_' training_data '_' testing_data '_' VOCCLASS{ii} '.txt'], 'w');
    model = models{ii};
    for jj=1:num_files
        filename = ['data/' net_model '/' testing_data '_' num2str(jj) '.mat'];
        disp(filename);
        load(filename);
        num_test = size(tests, 1);
        for kk=1:num_test
            disp(['Image ' num2str(kk)]);
            test = tests{kk};
            num_boxes = size(test, 1);
            features = zeros(num_boxes, 4096);
            boxes = zeros(num_boxes, 4);
            for bb=1:num_boxes
                features(bb, :) = test{bb}.feature';
                boxes(bb, :) = test{bb}.bbox;
            end
            features = features .* range;
            
            prediction = model.w * features';
            prediction = [-prediction; 1:num_boxes]';
            prediction = sortrows(prediction, 1);
            
            if -prediction(1, 1) > 0
                scores = -prediction(:, 1);
                index = scores > 0;
                scores = scores(index);
                dets = boxes(prediction(index, 2), :);

                [pruned_boxes, pruned_scores] = prune_detection(dets, scores);
                for bb=1:size(pruned_boxes, 1)
                    id = test_imgs{(jj-1)*img_per_file + kk};
                    box = pruned_boxes(bb, :);
                    line = [id ' ' num2str(pruned_scores(bb), 6) ' ' num2str(box(2)) ' ' num2str(box(1)) ' ' num2str(box(4)) ' ' num2str(box(3)) '\n'];
                    disp(line);
                    fprintf(fid, line);
                end
            end
        end
    end
    fclose(fid);
end