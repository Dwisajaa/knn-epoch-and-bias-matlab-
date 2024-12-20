function model = knnclassification_cornerfeature(train_feats, train_labels, max_epoch, threshold)
    % Inisialisasi model sebagai struct
    model.max_epoch = max_epoch;
    model.threshold = threshold;
    model.train_feats = train_feats;
    model.train_labels = train_labels;
end
