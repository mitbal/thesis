function detect(im, model, cls, id)

    boxes = selective_search(im);
    num_boxes = size(boxes, 1);
    
    features = zeros(4096, 1);
    
    for ii=1:num_boxes
        box = boxes(ii, :);
        region = im(box(1):box(3), box(2):box(4), :);
        rep = extract_caffe_feature(region);
        features(:, ii) = mean(rep, 2);
    end
    
    prediction = model.w * features;
    prediction = [-prediction; 1:num_boxes]';
    prediction = sortrows(prediction, 1);
    
    % Get the top region
    box = boxes(prediction(1, 2), :);
    score = -prediction(1, 1);
    disp(['score: ' num2str(score)]);
    if score > 0
        shapeInserter = vision.ShapeInserter('BorderColor', 'Custom', ...
            'CustomBorderColor', uint8([255 0 0]));
        rectangle = int32([box(2) box(1) box(4)-box(2)+1 box(3)-box(1)+1]);
        J = step(shapeInserter, im, rectangle);
        K = insertText(J, [box(2) box(1)], score, 'TextColor', [255 255 255], ...
            'BoxColor', [255 0 0]);
        imwrite(K, ['output/' num2str(cls) '_' num2str(id) '.png']);
%     imshow(K);
%     pause(0.1);
    end
    
end
