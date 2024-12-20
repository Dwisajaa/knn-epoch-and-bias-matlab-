% Pastikan Anda memiliki semua fungsi yang diperlukan
% Misalnya, `construct_vocabulary`, `BagsOfVisualWord`, `validate_model`, dan `classify_model`

% 1. Melatih model dengan minimal 3 kombinasi hyperparameter (k dan fungsi distance)

% Kombinasi pertama
k1 = 3;
dist1 = 'cosine';
vocab1 = construct_vocabulary(train_image_paths, k1, dist1);
train_feats1 = BagsOfVisualWord(train_image_paths, vocab1, dist1);

% Kombinasi kedua
k2 = 5;
dist2 = 'euclidean';
vocab2 = construct_vocabulary(train_image_paths, k2, dist2);
train_feats2 = BagsOfVisualWord(train_image_paths, vocab2, dist2);

% Kombinasi ketiga
k3 = 7;
dist3 = 'cityblock';
vocab3 = construct_vocabulary(train_image_paths, k3, dist3);
train_feats3 = BagsOfVisualWord(train_image_paths, vocab3, dist3);

% 2. Validasi model untuk setiap kombinasi hyperparameter dan catat akurasi
accuracy1 = validate_model(train_feats1, train_labels);
accuracy2 = validate_model(train_feats2, train_labels);
accuracy3 = validate_model(train_feats3, train_labels);

% 3. Catat akurasi dari setiap kombinasi hyperparameter
accuracies = [
    k1, dist1, accuracy1;
    k2, dist2, accuracy2;
    k3, dist3, accuracy3;
];

disp('Tabel Akurasi untuk Setiap Kombinasi Hyperparameter:');
disp('k   Distance    Akurasi');
disp(accuracies);

% 4. Pilih hyperparameter dengan akurasi terbaik
[~, best_idx] = max([accuracy1, accuracy2, accuracy3]); % Menemukan indeks dengan akurasi terbaik

% Tentukan hyperparameter terbaik berdasarkan akurasi tertinggi
switch best_idx
    case 1
        best_k = k1;
        best_dist = dist1;
    case 2
        best_k = k2;
        best_dist = dist2;
    case 3
        best_k = k3;
        best_dist = dist3;
end

% 5. Testing dengan kombinasi hyperparameter terbaik
vocab_best = construct_vocabulary(test_image_paths, best_k, best_dist);
test_feats = BagsOfVisualWord(test_image_paths, vocab_best, best_dist);

% Menggunakan hyperparameter terbaik
test_labels = test_image_labels; % Gambar yang tidak digunakan untuk training
predictions = classify_model(test_feats, vocab_best);

% 6. Catat hasil prediksi untuk 10 gambar yang diuji
disp('Hasil Prediksi untuk 10 Gambar Uji:');
for i = 1:10
    fprintf('Gambar %d Prediksi: %s\n', i, predictions{i});
end

% Fungsi untuk menghitung akurasi
function accuracy = validate_model(features, labels)
    % Misalnya menggunakan model k-NN untuk klasifikasi
    predicted_labels = knn_classifier(features); % Fungsi klasifikasi k-NN
    accuracy = sum(strcmp(predicted_labels, labels)) / numel(labels) * 100;
end

% Fungsi untuk klasifikasi menggunakan k-NN
function predicted_labels = knn_classifier(features)
    % Menggunakan k-NN dengan k=3 sebagai contoh
    k = 3;
    % Fungsionalitas k-NN dapat diubah sesuai kebutuhan
    predicted_labels = knnsearch(features, features, 'K', k); % Penggunaan fungsi k-NN untuk pencarian
end

% Fungsi untuk konstruksi vocabulary
function vocab = construct_vocabulary(image_paths, k, dist_type)
    % Kode untuk membangun vocabulary dengan k clusters dan tipe distance tertentu
    % Gantilah dengan kode yang sesuai untuk proyek Anda
    % Kode ini menggunakan k-means untuk clustering
    features = extract_features(image_paths);
    vocab = kmeans(features, k, 'Distance', dist_type); % K-means dengan jenis distance
end

% Fungsi untuk ekstraksi fitur gambar
function features = extract_features(image_paths)
    % Fungsi untuk mengekstrak fitur dari gambar-gambar
    % Misalnya, menggunakan SIFT, SURF, atau deskriptor lainnya
    % Gantilah dengan metode ekstraksi fitur yang sesuai
    features = [];
    for i = 1:numel(image_paths)
        img = imread(image_paths{i});
        % Ekstraksi fitur dari gambar, misalnya menggunakan SURF atau SIFT
        descriptor = extract_sift_features(img); % Ganti dengan fungsi ekstraksi fitur yang sesuai
        features = [features; descriptor]; % Gabungkan fitur yang diekstrak
    end
end

% Fungsi untuk ekstraksi fitur SIFT (contoh)
function descriptor = extract_sift_features(img)
    % Ekstraksi fitur SIFT dari gambar (gunakan SURF jika SIFT tidak tersedia)
    points = detectSURFFeatures(img);
    [features, valid_points] = extractFeatures(img, points);
    descriptor = features.Features;
end
