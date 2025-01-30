% ============================Measured=============================================
% Script Name: Brain Data Polynomial Dictionary Learning
% Description: This script implements polynomial dictionary learning on
%              brain fMRI data. It processes the data, constructs a 
%              3D graph, and learns a polynomial dictionary to analyze 
%              and represent the signals efficiently.
% Authors: Avi Halevi & Ben Moyal
% 
% Usage:
%   1. Ensure that the data file ('brain_data.csv') exists in 
%      the working directory.
%   2. Run the script directly in MATLAB.
%   3. Modify parameters (e.g., degree, sparsity level) as needed in the 
%      parameters section.
% 
% Outputs:
%   - Representation error as a function of the number of atoms used.
%   - Learned kernels of the polynomial dictionary.
%   - Most frequently used atoms visualized on the 3D brain graph.
%   - Visualization of accumulated signal strength across all vertices.
%   - Execution time summary.
% =========================================================================

clear; close all; clc;

% Start measuring execution time
start_time = tic;

%% Adding the paths
basePath = 'C:\GeometricsFinalProjectAviBen';
addpath(basePath);

% Load the data
data = readtable('brain_data.csv');

%% Filter data to a subset of 88 points

% Identify unique coordinates
unique_coords = unique([data.X, data.Y, data.Z], 'rows');

% Randomly select 88 unique points
rng(1); % Set random seed for reproducibility
selected_indices = randperm(size(unique_coords, 1), 1000);
selected_coords = unique_coords(selected_indices, :);

% Filter the data to include only the selected coordinates
is_selected = ismember([data.X, data.Y, data.Z], selected_coords, 'rows');
data = data(is_selected, :);

% Update unique_coords to reflect the filtered points
unique_coords = selected_coords;

%% Data preparation

%unique_coords = unique([data.X, data.Y, data.Z], 'rows');
N = size(unique_coords, 1);

% Normalize signal values
data.BOLD_Value = normalize(data.BOLD_Value, 'range');

% Split data into training and testing based on subjects
unique_subjects = unique(data.Subject);
Subjects = length(unique_subjects);
train_subjects = unique_subjects(1:round(0.6 * Subjects));
test_subjects = unique_subjects(round(0.6 * Subjects) + 1:end);

% Filter training and testing data
train_data = data(ismember(data.Subject, train_subjects), :);
test_data = data(ismember(data.Subject, test_subjects), :);

% Generate training signals
TrainSignal = zeros(N, height(train_data));
for i = 1:height(train_data)
    [~, node_idx] = ismember([train_data.X(i), train_data.Y(i), train_data.Z(i)], unique_coords, 'rows');
    TrainSignal(node_idx, i) = train_data.BOLD_Value(i);
end

% Generate testing signals
TestSignal = zeros(N, height(test_data));
for i = 1:height(test_data)
    [~, node_idx] = ismember([test_data.X(i), test_data.Y(i), test_data.Z(i)], unique_coords, 'rows');
    TestSignal(node_idx, i) = test_data.BOLD_Value(i);
end

%% Parameters
param.N = N;
param.S = 2;
degree = 5;
param.K = degree * ones(1, param.S);
param.T0 = 6; % Sparsity level
param.mu = 1e-4;
param.c = 1;
param.epsilon = 0.01;
param.numIteration = 4;
param.InitializationMethod = 'Random_kernels';
param.quadratic = 0;
param.displayProgress = 0;
param.plot_kernels = 0;
param.J = param.N * param.S;
param.thresh = 0.001;

%% Calculation of W

distances = pdist2(unique_coords, unique_coords);

% Create the weight matrix W
W = zeros(N, N);
mask = distances > 0 & distances < 8; % 8 mm
W(mask) = 1 ./ distances(mask);

% Plot the matrix W
figure;
imagesc(W); % Display the similarity matrix
colorbar; % Add a colorbar for scale
imagesc(log10(W + 1e-6));
title('Matrix W (Log Scale)');
xlabel('Node Index');
ylabel('Node Index');
axis square; % Make the plot square

% Check if the graph is connected
if ~all(conncomp(graph(W)) == 1)
    error('The graph is not connected!');
end

%% Compute the Laplacian and the normalized Laplacian operator
L = diag(sum(W, 2)) - W; % Combinatorial Laplacian
param.Laplacian = (diag(sum(W, 2)))^(-1/2) * L * (diag(sum(W, 2)))^(-1/2); % Normalized Laplacian
[param.eigenMat, param.eigenVal] = eig(param.Laplacian); % Eigendecomposition of the normalized Laplacian
[param.lambda_sym, index_sym] = sort(diag(param.eigenVal)); % Sort eigenvalues in ascending order

%% Compute the powers of the Laplacian
for k = 0 : max(param.K) % Ensure we go up to K+1 powers
    param.Laplacian_powers{k + 1} = param.Laplacian^k; % k+1 because indexing starts at 1
end

% Compute Vandermonde matrix for eigenvalues
for j = 1:param.N
    for i = 0:max(param.K) % Ensure we go up to K+1 powers
        param.lambda_powers{j}(i + 1) = param.lambda_sym(j)^(i);
        param.lambda_power_matrix(j, i + 1) = param.lambda_sym(j)^(i); % Build Vandermonde matrix
    end
end
%% Dictionary learning

% Run the dictionary learning algorithm
[recovered_Dictionary, output, g_ker_learned] = Polynomial_Dictionary_Learning(TrainSignal, param);

% Compute representation error for different numbers of atoms
%num_atoms = [1, 5, 10, 15, 20];
num_atoms = 1:25;
errors = zeros(length(num_atoms), 1);
atom_usage = zeros(param.J, 1);


for idx = 1:length(num_atoms)
    T = num_atoms(idx); % Number of atoms used
    CoefMatrix = OMP_non_normalized_atoms(recovered_Dictionary, TestSignal, T); % Sparse coding
    atom_usage = atom_usage + sum(abs(CoefMatrix) > 1e-6, 2);
    reconstructed = recovered_Dictionary * CoefMatrix;
    errors(idx) = compute_average_normalized_error(TestSignal, reconstructed);
end

% Plot the representation error
figure;
plot(num_atoms, errors, '-o', 'LineWidth', 1.5, 'MarkerSize', 6);
title('Representation Error vs. Number of Atoms');
xlabel('Number of Atoms Used in the Representation');
ylabel('Average Normalized Representation Error');
grid on;


% Plot the learned kernels
figure;
hold on;
for s = 1:param.S
    plot(param.lambda_sym, g_ker_learned(:, s), '-o', ...
        'LineWidth', 1.5, 'MarkerSize', 3);
end
title('Learned Kernels', 'FontSize', 14);
xlabel('Eigenvalues of the Laplacian (\lambda)', 'FontSize', 12);
ylabel('Generating kernels  $\hat{g}(\lambda)$', 'Interpreter', 'latex');
grid on;
hold off;

%% Identify and plot the 6 most used atoms
[~, most_used_indices] = maxk(atom_usage, 6);



for i = 1:length(most_used_indices)
    figure;
    scatter3(unique_coords(:, 1), unique_coords(:, 2), unique_coords(:, 3), 30, [0.7, 0.7, 0.7], 'filled');
    hold on;
    atom_idx = most_used_indices(i);
    weights = abs(recovered_Dictionary(:, atom_idx));
    scatter3(unique_coords(:, 1), unique_coords(:, 2), unique_coords(:, 3), 25, weights, 'filled');
    colorbar;
    title('Most Used Atoms on the Brain Graph');
    xlabel('X (mm)');
    ylabel('Y (mm)');
    zlabel('Z (mm)');
    grid on;
    axis equal;
    hold off;
end



%% Accumulate total signals
[~, node_indices] = ismember([data.X, data.Y, data.Z], unique_coords, 'rows');
total_BOLD_Value = accumarray(node_indices, data.BOLD_Value, [N, 1], @sum, 0);

figure;
scatter3(unique_coords(:, 1), unique_coords(:, 2), unique_coords(:, 3), 30, total_BOLD_Value, 'filled');
colorbar;
title('Total BOLD_Value Signals Across All Nodes');
xlabel('X (mm)');
ylabel('Y (mm)');
zlabel('Z (mm)');
grid on;
axis equal;

%% Runtime calculation
elapsed_time = toc(start_time);
fprintf('Execution time: %.2f seconds\n', elapsed_time);
