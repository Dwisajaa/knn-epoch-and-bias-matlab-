clc;
clear;

% Set parameters
data_path = 'C:/deteksiknn/';  % Path to your dataset
categories = {'n02093647-Bedlington_terrier', 'n02093754-Border_terrier'};
num_train_per_cat = 100;
num_valid_per_cat = 40;
vocab_size = 128;

% Get image paths and labels
[train_image_paths, valid_image_paths, train_labels, valid_labels] = ...
    get_image_paths(data_path, categories, num_train_per_cat, num_valid_per_cat);

% Construct vocabulary
vocab = construct_vocabulary(train_image_paths, vocab_size);

% Extract features
train_feats = BagsOfVisualWord(train_image_paths, vocab);
valid_feats = BagsOfVisualWord(valid_image_paths, vocab);

% Define hyperparameter ranges
epoch_values = [50, 100, 150];  % Maximum number of epochs
bias_values = [0.1, 0.5, 1.0];  % Bias or threshold values
accuracies = zeros(length(epoch_values), length(bias_values));
models = cell(length(epoch_values), length(bias_values));

% Train and evaluate for different hyperparameter combinations
for i = 1:length(epoch_values)
    for j = 1:length(bias_values)
        % Train model
        models{i, j} = knnclassification_cornerfeature(train_feats, train_labels, ...
            epoch_values(i), bias_values(j));
        
        % Predictions for validation set
        pred_labels = testing(models{i, j}, valid_feats);
        accuracies(i, j) = mean(strcmp(pred_labels, valid_labels));
        
        % Print predictions for each combination
        fprintf('\nPredictions for Epoch = %d, Bias = %.1f:\n', epoch_values(i), bias_values(j));
        disp(pred_labels');
    end
end

% Find the best model based on validation accuracy
[best_accuracy, best_idx] = max(accuracies(:));
[best_epoch_idx, best_bias_idx] = ind2sub(size(accuracies), best_idx);
best_epoch = epoch_values(best_epoch_idx);
best_bias = bias_values(best_bias_idx);
best_model = models{best_epoch_idx, best_bias_idx};

% Print the best model's details
fprintf('\nBest Model Based on Validation Accuracy:\n');
fprintf('Epoch: %d\n', best_epoch);
fprintf('Bias: %.1f\n', best_bias);
fprintf('Validation Accuracy: %.2f%%\n', best_accuracy * 100);

% Test the best model
[test_image_paths, ~] = get_image_paths(data_path, categories, 5, 0);  % Get test set with 5 images
test_feats = BagsOfVisualWord(test_image_paths, vocab);
test_predictions = testing(best_model, test_feats);

% Print test predictions
disp('\nTest Predictions (Best Model):');
disp(test_predictions);

% File untuk menyimpan hasil prediksi
output_file = 'test_predictions_best.txt';
fid = fopen(output_file, 'w');

% Menyimpan hasil prediksi untuk model terbaik
fprintf(fid, '\nBest Model Predictions (Epoch = %d, Bias = %.1f):\n', best_epoch, best_bias);
for i = 1:length(test_predictions)
    fprintf(fid, '%s\n', test_predictions{i});
end
% Print all validation accuracies
fprintf('\nValidation Accuracies for All Hyperparameter Combinations:\n');
disp(array2table(accuracies, 'RowNames', ...
    cellfun(@(x) sprintf('Epoch-%d', x), num2cell(epoch_values), 'UniformOutput', false), ...
    'VariableNames', ...
    cellfun(@(x) sprintf('Bias-%.1f', x), num2cell(bias_values), 'UniformOutput', false)));

% Test all models
[test_image_paths, ~] = get_image_paths(data_path, categories, 5, 0);  % Get test set with 5 images
test_feats = BagsOfVisualWord(test_image_paths, vocab);

fprintf('\nTest Predictions for All Hyperparameter Combinations:\n');
output_file = 'test_predictions_all.txt';
fid = fopen(output_file, 'w');

for i = 1:length(epoch_values)
    for j = 1:length(bias_values)
        fprintf(fid, '\nPredictions for Epoch = %d, Bias = %.1f:\n', epoch_values(i), bias_values(j));
        fprintf('\nPredictions for Epoch = %d, Bias = %.1f:\n', epoch_values(i), bias_values(j));
        
        % Test predictions
        test_preds = testing(models{i, j}, test_feats);
        disp(test_preds);
        
        % Save predictions to file
        for k = 1:length(test_preds)
            fprintf(fid, '%s\n', test_preds{k});
        end
    end
end
fclose(fid);
