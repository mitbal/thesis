function detect(im, models, id, VOCCLASS)

    boxes = selective_search(im);
    num_boxes = size(boxes, 1);
    
    features = zeros(4096, 1);
    
    for ii=1:num_boxes
        box = boxes(ii, :);
        region = im(box(1):box(3), box(2):box(4), :);
        rep = extract_caffe_feature(region);
        features(:, ii) = mean(rep, 2);
    end
    
    for ii=1:20
        disp(['Class: ' VOCCLASS{ii}]);
        model = models{ii};
        prediction = model.w * features;
        prediction = [-prediction; 1:num_boxes]';
        prediction = sortrows(prediction, 1);

        % Get all positive detection, start from the top
        mark = im(:,:,:);
        for jj=1:num_boxes
            box = boxes(prediction(jj, 2), :);
            score = -prediction(jj, 1);
            if score > 0
                disp(['score: ' num2str(score)]);
                shapeInserter = vision.ShapeInserter('BorderColor', 'Custom', ...
                    'CustomBorderColor', uint8([255 0 0]));
                rectangle = int32([box(2) box(1) box(4)-box(2)+1 box(3)-box(1)+1]);
                mark = step(shapeInserter, mark, rectangle);
                mark = insertText(mark, [box(2) box(1)], score, 'TextColor', [255 255 255], ...
                    'BoxColor', [255 0 0]);
            else
                break;
            end
        end
        if -prediction(1, 1) > 0
            imwrite(mark, ['output/' num2str(id) '_' VOCCLASS{ii} '.png']);
        end
    end
end
