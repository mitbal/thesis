function features = extract_caffe_feature(patch)
    
    model_file = 'models/rcnn/finetune_voc_2007_trainval_iter_100000.caffemodel';
    def_file = 'models/rcnn/deploy.prototxt';
    
    if caffe('is_initialized') == 0
        caffe('init', def_file, model_file);
        caffe('set_phase_test');
        caffe('set_device', 0);
        caffe('set_mode_gpu');
    end
    
    images = {prepare_image(patch)};
    features = caffe('forward', images);
    features = permute(features{1}, [3 4 1 2]);
end