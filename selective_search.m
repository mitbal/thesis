function boxes = selective_search(im)

    % set color space
    colorTypes = {'Hsv', 'Lab', 'RGI', 'H', 'Intensity'};
    colorTypes = colorTypes{1};
    
    % set similarity measure
    simFunctionHandles = {@SSSimColourTextureSizeFillOrig, @SSSimTextureSizeFill, @SSSimBoxFillOrig, @SSSimSize};
    simFunctionHandles = simFunctionHandles(1:2);
    
    % parameter for Felzenszwalb graph based segmentation algorithm
    k = 200;        % lower k gives more regions
    minSize = k;
    sigma = 0.8;
    
    [boxes, ~, ~, ~] = Image2HierarchicalGrouping(im, sigma, k, minSize, colorTypes, simFunctionHandles);
    boxes = BoxRemoveDuplicates(boxes);

end
