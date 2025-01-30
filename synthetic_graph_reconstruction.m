% =========================================================================
% Script Name: synthetic_graph_reconstruction
% Description: This script generates synthetic graph signals, trains a
%              polynomial dictionary, and evaluates the learned dictionary.
%              The script follows the methodology in the paper:
%              "Learning Parametric Dictionaries for Signals on Graphs"
% Authors: Avi Halevi & Ben Moyal
% 
% Usage:
%   1. Run this script directly in MATLAB.
%   2. Modify parameters at the beginning of the script as needed.
% 
% Outputs:
%   - Original and learned kernels are plotted.
%   - Training and testing errors are displayed.
% 
% =========================================================================

clear; close all; clc;
rng(1); % Set random seed for reproducibility

%% Adding the paths
basePath = 'C:\Users\Avi\Documents\MATLAB\Edited_parametric_learning_code';
addpath(basePath);
load('C.mat')
load('Article_W.mat')

%% Parameters
% =======================
% Parameters explicitly mentioned in the paper
param.N = 100;           % Number of vertices
param.S = 4;             % Number of subdictionaries
degree = 5;

param.K = degree * ones(1, param.S); % Polynomial degree per subdictionary

param.T0 = 4; % Sparsity level in the training phase
theta = 0.9;       % Gaussian kernel parameter
kappa = 0.2;       % Threshold distance. Adjusted from the paper, possible typo in the paper.A.H
param.nb_train_signals = 600; % Number of training signals
param.nb_test_signals = 2000; % Number of testing signals
param.mu = 1e-4; % Polynomial regularizer parameter

% Optimization-related parameters (not explicitly mentioned in the paper)
param.c = 1; % Spectral control parameter
param.epsilon = 0.02; % We assume that epsilon_1 = epsilon_2 = epsilon.
%param.numIteration = 25;  % Number of iterations for dictionary learning
param.numIteration = 4;  % Number of iterations for dictionary learning
param.InitializationMethod =  'Random_kernels'; % Dictionary initiainit_alphalization method
param.plot_kernels = 0;  % Disable kernel plotting during learning
param.quadratic = 0; % solve the quadratic program using interior point methods
param.displayProgress = 0;

% Derived parameters
param.J = param.N * param.S; % Total number of atoms
param.thresh = 0.001;

%% Signal generation
isConnected = false;
attempt = 0;
while ~isConnected
    attempt = attempt + 1;
    points = rand(param.N, 2); % N points with x and y coordinates
    distances = pdist2(points, points); % Pairwise distance matrix
    W = exp(-(distances.^2) / (2 * theta^2)); % Gaussian similarity
    W(distances >= kappa) = 0; % Apply thresholding to sparsify W
    graphObj = graph(W); % Create graph object
    isConnected = all(conncomp(graphObj) == 1); % Check connectivity
    if ~isConnected
        fprintf('Attempt %d: Graph is not connected, retrying...\n', attempt);
    end
end

% Plot the points on the unit square
figure;
scatter(points(:, 1), points(:, 2), 50, 'filled'); % Scatter plot of points
title('Random Points in the Unit Square');
xlabel('X-coordinate');
ylabel('Y-coordinate');
grid on;

% Plot the matrix W
figure;
imagesc(W); % Display the similarity matrix
colorbar; % Add a colorbar for scale
title('Matrix W (Similarity Matrix)');
xlabel('Node Index');
ylabel('Node Index');
axis square; % Make the plot square

% Plot the Matrix W attached to the articl
figure;
imagesc(Article_W); % Display the similarity matrix
colorbar; % Add a colorbar for scale
title('Matrix W attached to the article');
xlabel('Node Index');
ylabel('Node Index');
axis square; % Make the plot square

% For kappa = 0.5
isConnected = false;
while ~isConnected  
    points = rand(param.N, 2); % N points with x and y coordinates
    distances = pdist2(points, points); % Pairwise distance matrix
    W2 = exp(-(distances.^2) / (2 * theta^2)); % Gaussian similarity
    W2(distances >= 0.5) = 0; % Apply thresholding to sparsify W
    graphObj = graph(W2); % Create graph object
    isConnected = all(conncomp(graphObj) == 1); % Check connectivity
end

% Plot the matrix W for kappa = 0.5
figure;
imagesc(W2); % Display the similarity matrix
colorbar; % Add a colorbar for scale
title('Matrix W for kappa = 0.5');
xlabel('Node Index');
ylabel('Node Index');
axis square; % Make the plot square


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

init_alpha = C;

%% Generate Polynomial Generating Dictionary
D = zeros(param.N, param.J);
r = 0;
for i = 1 : param.S
    D_sub = zeros(param.N, param.N);
    for l = 0 : param.K(i)
        D_sub = D_sub +  init_alpha(l + 1 + r)*param.Laplacian_powers{l + 1};
    end
    D(:, (param.N*(i-1)) + 1 : param.N*i) = D_sub;
    r = sum(param.K(1:i)) + i;
end

%% Create TestSignal
X = zeros(param.J, param.nb_test_signals); 
for i = 1:param.nb_test_signals
    atom_indices = randperm(param.J, param.T0);
    X(atom_indices, i) = randn(param.T0, 1);
end
TestSignal = D * X;

%% Signal-to-Noise Ratio (SNR) Calculation for Learned Kernels

% Define M values (sizes of training sets)
M_values = [400, 600, 2000];
SNR_values = zeros(length(M_values), 1); % To store SNR results
g_kernels_learned = cell(length(M_values), 1); % Store learned kernels for each M
num_atoms = 1:25; % Range of number of atoms
errors_per_M = cell(length(M_values), 1); % To store errors for each M


% Compute the original kernels (g_b^s_0(Λ))
g_ker_original = zeros(param.N, param.S);
r = 0;
for i = 1 : param.S
    for n = 1 : param.N
    p = 0;
    for l = 0 : param.K(i)
        p = p +  init_alpha(l + 1 + r)*param.lambda_powers{n}(l + 1);
    end
    g_ker_original(n,i) = p;
    end
    r = sum(param.K(1:i)) + i;
end


% Plot original kernels
figure;
hold on;
for s = 1:param.S
    plot(param.lambda_sym, g_ker_original(:, s), '-o', 'LineWidth', 1.5, ...
         'MarkerSize', 3); % Use consistent colors
end
title('Kernels of the generating dictionary');
xlabel('Eigenvalues of the Laplacian (\lambda)');
ylabel('Generating kernels  $\hat{g}(\lambda)$', 'Interpreter', 'latex');
grid on;
hold off;

%%

for i = 1:length(M_values)
    M = M_values(i);
    fprintf('\nProcessing M = %d training signals...\n', M);

    % Update the number of training signals
    param.nb_train_signals = M;

    % Generate new training signals
    TrainSignal = generate_signals(D, param);

    % Learn the dictionary
    [recovered_Dictionary, ~, g_ker_learned] = Polynomial_Dictionary_Learning(TrainSignal, param);

    % Store the learned kernels
    g_kernels_learned{i} = g_ker_learned;

    % Compute the mean SNR for this M
    SNR_values(i) = compute_mean_snr(g_ker_original, g_ker_learned, param);
    % Plot the learned kernels
    figure;
    hold on;
    for s = 1:param.S
        plot(param.lambda_sym, g_ker_learned(:, s), '-o', ...
            'LineWidth', 1.5, 'MarkerSize', 3);
    end
    title(sprintf('Learned Kernels with (M = %d)', M), 'FontSize', 14);
    xlabel('Eigenvalues of the Laplacian (\lambda)', 'FontSize', 12);
    ylabel('Generating kernels  $\hat{g}(\lambda)$', 'Interpreter', 'latex');
    grid on;
    hold off;

    % Compute representation error for different numbers of atoms
    errors = zeros(length(num_atoms), 1); % Initialize errors for this M

    for idx = 1:length(num_atoms)
        T = num_atoms(idx); % Number of atoms used
        CoefMatrix = OMP_non_normalized_atoms(recovered_Dictionary, TestSignal, T); % Sparse coding
        reconstructed = recovered_Dictionary * CoefMatrix; % Reconstruct signals
        errors(idx) = compute_average_normalized_error(TestSignal, reconstructed); % Compute error
    end

    % Store the errors for this M
    errors_per_M{i} = errors;

    % Plot representation error for this M
    figure;
    plot(num_atoms, errors, '-o', 'LineWidth', 1.5, 'MarkerSize', 6);
    title(sprintf('Representation Error vs. Number of Atoms (M = %d)', M));
    xlabel('Number of Atoms Used in the Representation');
    ylabel('Average Normalized Representation Error');
    grid on;    

end

fprintf('\nFinal SNR Results:\n');
for i = 1:length(M_values)
    fprintf('M = %d: SNR builtin = %.2f dB\n', M_values(i), SNR_values(i));
end

%% Function to compute the mean SNR
function mean_snr = compute_mean_snr(g_ker_original, g_ker_learned, param)
    % Extract eigenvector matrix (χ) from param
    chi = param.eigenMat; % Eigenvector matrix (χ)
    S = param.S;
    D_original = cell(1, S);
    D_learned = cell(1, S);
    
    for s = 1:S
        D_original{s} = chi * diag(g_ker_original(:, s)) * chi';
        D_learned{s} = chi * diag(g_ker_learned(:, s)) * chi';
    end

    snr_values = zeros(S, 1);
    for s = 1:S
        error_norm = norm(D_original{s} - D_learned{s}, 2); % Frobenius norm
        snr_values(s) = -20 * log10(error_norm); % SNR in dB
    end

    mean_snr = mean(snr_values);
end



%% Function to create TrainSignal
function [TrainSignal] = generate_signals(D, param)
    X = zeros(param.J, param.nb_train_signals);
    for i = 1:param.nb_train_signals
        atom_indices = randperm(param.J, param.T0);
        X(atom_indices, i) = randn(param.T0, 1);
    end
    TrainSignal = D * X;
end
