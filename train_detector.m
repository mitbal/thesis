% Train SVM detector
init;

% Load positive data
pos = load('data/baseline_positive.mat');
num_pos = size(pos.labels, 1);

% Load the first negative data
% num_neg = 0;
% neg_features = zeros(700000, 4096);
% for ii=1:6
%     disp(['Negative data: ' num2str(ii)]);
%     neg = load(['data/baseline_negative_' num2str(ii) '.mat']);
%     for jj=1:size(neg.features, 2)
%         neg_features(jj+num_neg, :) = neg.features{jj}';
%     end
%     num_neg = num_neg + size(neg.features, 2);
% end
neg = load('data/baseline_negative_1.mat');
num_neg = size(neg.features, 2);

% Construct 1 big matrix
num_train = num_pos+num_neg;
features = zeros(num_train, 4096);
for ii=1:num_pos
    features(ii, :) = pos.features{ii}';
end
for ii=1:num_neg
    features(ii+num_pos, :) = neg.features{ii}';
end
labels = cell2mat(pos.labels);
labels = [labels; zeros(num_neg, 1)];

% Normalization??? Skip for now

% Train detector for each class
models = cell(1);
for ii=1:20
    index = labels == ii;
    train_features = [features(index, :); features((labels == 0), :)];
    train_labels = [ones(sum(index), 1); -ones(num_neg, 1)];
    
    disp(['Training class: ' num2str(ii)]);
    train_features = sparse(train_features);
    model = train(train_labels, train_features, '-c 1 -s 2');
    models{ii} = model;
end

save('svm_models/baseline.mat', 'models', '-v7.3');
