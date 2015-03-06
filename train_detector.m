% Train SVM detector
clear all;
init;

net_model = 'caffenet';
pos_dataset = 'trainval';
dataset = 'train';

% Load positive data
pos = load(['data/' net_model '/' pos_dataset '_positive.mat']);
num_pos = size(pos.labels, 2);

% Load the first negative data
neg = load(['data/' net_model '/' dataset '_negative_1.mat']);
num_neg = size(neg.features, 2);

% Construct 1 big matrix
num_train = num_pos+num_pos;
features = zeros(num_train, 4096);
for ii=1:num_pos
    features(ii, :) = pos.features{ii}';
end
for ii=1:num_pos
    features(ii+num_pos, :) = neg.features{ii}';
end
labels = cell2mat(pos.labels);
labels = [labels'; zeros(num_pos, 1)];

% Normalization??? Skip for now


% Train detector for each class
models = cell(20,1);
for ii=1:20
    index = labels == ii;
    num_pos = sum(index);
    num_neg = num_pos;
    neg_features = features(labels==0, :);
    features_cache = [features(index, :); neg_features(1:num_pos, :)];
    train_labels = [ones(sum(index), 1); -ones(num_pos, 1)];
    
    disp(['Training class: ' num2str(ii)]);
    train_features = sparse(features_cache);
    options = ['-w1 ' num2str(2) ' -c ' num2str(0.001) ' -s 2']; % Parameters taken from rcnn code
    model = train(train_labels, train_features, options);
    
    scores = model.w * train_features';
    prediction = scores > 0;
    prediction = prediction .* 2 -1;
    prediction = prediction';
    tp = sum(prediction(1:num_pos)==train_labels(1:num_pos));
    tn = sum(prediction(num_pos+1:end)==train_labels(1+num_pos:end));
    disp(['tp ' num2str(tp) '/' num2str(num_pos)]);
    disp(['tn ' num2str(tn) '/' num2str(num_neg)]);
    
    for jj=1:6
        disp(['Load the ' num2str(jj) ' negative data']);
        neg = load(['data/' net_model '/' dataset '_negative_' num2str(jj) '.mat']);
        num_neg = size(neg.features, 2);
        neg_features = zeros(num_neg, 4096);
        for kk=1:num_neg
            neg_features(kk, :) = neg.features{kk}';
        end
        
        % Get misclassified data
        % should be normalized first
        scores = model.w * neg_features';
        index = scores > 0;
        num_wrong = sum(index);
        disp(['wrong ' num2str(num_wrong)]);
        
        features_cache = [features_cache; neg_features(index, :)];
        train_features = sparse(features_cache);
        train_labels = [train_labels; -ones(num_wrong, 1)];
        num_neg = num_neg + num_wrong;
        
        % Retrain
        model = train(train_labels, train_features, options);
        
        % Evaluate
        scores = model.w * train_features';
        prediction = scores > 0;
        prediction = prediction .* 2 -1;
        prediction = prediction';
        tp = sum(prediction(1:num_pos)==train_labels(1:num_pos));
        tn = sum(prediction(num_pos+1:end)==train_labels(1+num_pos:end));
        disp(['tp ' num2str(tp) '/' num2str(num_pos)]);
        disp(['tn ' num2str(tn) '/' num2str(size(train_labels,1)-num_pos)]);
    end
    
    models{ii} = model;
    save(['svm_models/' net_model '/' dataset '.mat'], 'models', '-v7.3');
end

