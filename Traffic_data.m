% =========================================================================
% Script Name: Traffic Data Polynomial Dictionary Learning
% Description: This script implements polynomial dictionary learning on 
%              traffic delay data. It processes the data, constructs a 
%              graph, and learns a polynomial dictionary to analyze and 
%              represent the signals efficiently.
% Authors: Avi Halevi & Ben Moyal
% 
% Usage:
%   1. Ensure that the data file ('Partial_Traffic_data.csv') exists in 
%      the working directory.
%   2. Run the script directly in MATLAB.
%   3. Modify parameters (e.g., degree, sparsity level) as needed in the 
%      parameters section.
% 
% Outputs:
%   - Representation error as a function of the number of atoms used.
%   - Learned kernels of the polynomial dictionary.
%   - Most frequently used atoms visualized on the graph.
%   - Visualization of accumulated delays across all nodes in the graph.
%   - Execution time summary.
% =========================================================================


clear; close all; clc;

% Start measuring execution time
start_time = tic;


%% Adding the paths
basePath = 'C:\Users\Avi\Documents\MATLAB\Edited_parametric_learning_code';
addpath(basePath);

% Load the data
%data = readtable('Traffic_data.csv');
data = readtable('Partial_Traffic_data.csv');
%% Vertex sampling
% rng(1); % Set random seed for reproducibility
% % Identify unique coordinates
% unique_coords = unique([data.Start_Lat_Representative, data.Start_Lng_Representative], 'rows');
% 
% % Select 400 random indices
% selected_indices = randperm(size(unique_coords, 1), 440); % Randomly select 400 nodes
% selected_coords = unique_coords(selected_indices, :); % Extract the selected coordinates
% 
% % Filter the data to include only the selected coordinates
% is_selected = ismember([data.Start_Lat_Representative, data.Start_Lng_Representative], selected_coords, 'rows');
% data = data(is_selected, :); % Keep only the rows corresponding to the selected nodes
% 
% writetable(data, 'Partial_Traffic_data.csv');
% 
% return
%% Data preparation

unique_coords = unique([data.Start_Lat_Representative, data.Start_Lng_Representative], 'rows');
N = size(unique_coords, 1);

% Convert dates to MATLAB datetime format
data.Date = datetime(data.Date, 'InputFormat', 'dd/MM/yyyy');

% Sort data by date
data = sortrows(data, 'Date');

% Normalize signal values
data.DelayFromTypicalTrafficMins = normalize(data.DelayFromTypicalTrafficMins, 'range');

% Split into training and testing
num_entries = height(data);
split_index = round(num_entries * 0.6);
train_data = data(1:split_index, :);
test_data = data(split_index + 1:end, :);

% Generate training signals
TrainSignal = zeros(N, split_index);
for i = 1:split_index
    [~, node_idx] = ismember([train_data.Start_Lat_Representative(i), train_data.Start_Lng_Representative(i)], unique_coords, 'rows');
    TrainSignal(node_idx, i) = train_data.DelayFromTypicalTrafficMins(i);
end

% Generate testing signals
TestSignal = zeros(N, num_entries - split_index);
for i = 1:(num_entries - split_index)
    [~, node_idx] = ismember([test_data.Start_Lat_Representative(i), test_data.Start_Lng_Representative(i)], unique_coords, 'rows');
    TestSignal(node_idx, i) = test_data.DelayFromTypicalTrafficMins(i);
end

%% Parameters
% =======================
param.N = N;           % Number of vertices
param.S = 2;             % Number of subdictionaries
degree = 10;
param.K = degree * ones(1, param.S); % Polynomial degree per subdictionary
param.T0 = 6; % Sparsity level in the training phase
param.mu = 1e-4; % Polynomial regularizer parameter

% Optimization-related parameters
param.c = 1; % Spectral control parameter
param.epsilon = 0.01; % We assume that epsilon_1 = epsilon_2 = epsilon.
param.numIteration = 4;  % Number of iterations for dictionary learning
param.InitializationMethod =  'Random_kernels'; % Dictionary initiainit_alphalization method
param.quadratic = 0; % solve the quadratic program using interior point methods

% Preferences
param.displayProgress = 0;
param.plot_kernels = 0;  % Disable kernel plotting during learning

% Derived parameters
param.J = param.N * param.S; % Total number of atoms
param.thresh = 0.001;

%% Calculation of W

distances = pdist2(unique_coords, unique_coords);

% Create the weight matrix W
W = zeros(N, N); % Initialize weight matrix
%threshold = 6 / 111; % Convert 11 km to degrees
threshold = 7 / 111; % Convert 7 km to degrees
mask = distances > 0 & distances < threshold; % Logical mask for valid edges
W(mask) = 1 ./ distances(mask); % Assign inverse distances for valid edge

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

%% Graph Visualization
% Randomly sample a subset of nodes
num_nodes_to_keep = 400; % Adjust the number of nodes to keep
sampled_indices = randperm(size(unique_coords, 1), num_nodes_to_keep);

% Subset the coordinates and matrix W
sampled_coords = unique_coords(sampled_indices, :);
sampled_W = W(sampled_indices, sampled_indices);

% Extract coordinates for plotting
x_coords = sampled_coords(:, 2); % Longitude
y_coords = sampled_coords(:, 1); % Latitude

% Plot the edges
figure;
hold on;
[m, n] = size(sampled_W);
for i = 1:m
    for j = 1:n
        if sampled_W(i, j) > 0
            plot([x_coords(i), x_coords(j)], [y_coords(i), y_coords(j)], 'Color', '#198166');
        end
    end
end

% Set plot properties
title('Graph Visualization');
xlabel('Longitude');
ylabel('Latitude');
% grid on;
% axis equal;
hold off;

% Coordinates are not sampled for the continuation of the code
x_coords = unique_coords(:, 2); % Longitude
y_coords = unique_coords(:, 1); % Latitude

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
[~, most_used_indices] = maxk(atom_usage, 6); % Get the indices of the 6 most used atoms

% Create scatter plot for each of the 6 atoms
% Loop through the 6 most used atoms
for i = 1:length(most_used_indices)
    figure;
    scatter(x_coords, y_coords, 30, 'filled', 'MarkerFaceColor', [0.7, 0.7, 0.7]); % Background scatter
    hold on;
    atom_idx = most_used_indices(i);
    weights = abs(recovered_Dictionary(:, atom_idx)); % Extract weights for this atom

    % Plot scatter with color representing the weights
    scatter(x_coords, y_coords, 25, weights, 'filled');
    % Colorbar and axis labels
    colorbar;
    title('Most Used Atoms on the Graph');
    xlabel('Longitude');
    ylabel('Latitude');
    grid on;
    axis equal;
    hold off;
end


%% Accumulate total signals
% Map each row in the data table to its corresponding node index
[~, node_indices] = ismember([data.Start_Lat_Representative, data.Start_Lng_Representative], unique_coords, 'rows');

% Accumulate total delays for each unique node
total_delays = accumarray(node_indices, data.DelayFromTypicalTrafficMins, [N, 1], @sum, 0);

% Create a scatter plot with colors based on accumulate total delays
figure;
scatter(x_coords, y_coords, 30, total_delays, 'filled');

% Add colorbar and labels
colorbar;
title('Total Delays Across All Periods (Normalized)');
xlabel('Longitude');
ylabel('Latitude');
grid on;
axis equal;

%% Runtime calculation and printing
% Stop measuring execution time and calculate elapsed time
elapsed_time = toc(start_time);

% Convert elapsed time to hours, minutes, and seconds
hours = floor(elapsed_time / 3600);
minutes = floor(mod(elapsed_time, 3600) / 60);
seconds = mod(elapsed_time, 60);

% Display elapsed time in a human-readable format
fprintf('Execution time: %02d:%02d:%05.2f (hh:mm:ss)\n', hours, minutes, seconds);

