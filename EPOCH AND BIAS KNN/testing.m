function predictions = testing(model, test_feats)
    num_samples = size(test_feats, 1);
    predictions = cell(num_samples, 1);
    
    for i = 1:num_samples
        % Hitung jarak ke semua fitur pelatihan
        distances = sum(abs(model.train_feats - test_feats(i, :)), 2);
        
        % Ambil label dengan jarak minimum jika memenuhi threshold
        [min_distance, idx] = min(distances);
        if min_distance <= model.threshold
            predictions{i} = model.train_labels{idx};
        else
            predictions{i} = 'Unknown'; % Jika tidak memenuhi threshold
        end
    end
end
