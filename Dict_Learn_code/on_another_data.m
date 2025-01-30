clear; close all; clc;
%% Adding the paths

basePath = 'C:\Users\Avi\Documents\MATLAB\Edited_parametric_learning_code';
addpath(basePath);
addpath(fullfile(basePath, 'DataSets'));  %Folder conatining the yalmip tools
load('testdata.mat')


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
param.numIteration = 8;  % Number of iterations for dictionary learning
param.InitializationMethod =  'Random_kernels'; % Dictionary initiainit_alphalization method
param.plot_kernels = 0;  % Disable kernel plotting during learning
param.quadratic = 0; % solve the quadratic program using interior point methods
param.displayProgress = 0;

% Derived parameters
param.J = param.N * param.S; % Total number of atoms
param.thresh = 0.001;

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
        param.lambda_powers{j}(i + 1) = param.lambda_sym(j)^(i); % שימוש פנימי בלולאה A.H
        param.lambda_power_matrix(j, i + 1) = param.lambda_sym(j)^(i); % Build Vandermonde matrix
    end
end
init_alpha = C;

%% הגדרת פונקציה ליצירת אותות אימון
function [TrainSignal, TestSignal] = generate_signals(D, param)
    % TrainSignal
    X = zeros(param.J, param.nb_train_signals);
    for i = 1:param.nb_train_signals
        atom_indices = randperm(param.J, param.T0); % T0 אינדקסים אקראיים
        X(atom_indices, i) = randn(param.T0, 1); % ערכים אקראיים
    end
    TrainSignal = D * X; % מכפלה בין המילון למטריצה X

    % TestSignal
    X = zeros(param.J, param.nb_test_signals); 
    for i = 1:param.nb_test_signals
        atom_indices = randperm(param.J, param.T0); % T0 אינדקסים אקראיים
        X(atom_indices, i) = randn(param.T0, 1); % ערכים אקראיים
    end
    TestSignal = D * X; % מכפלה בין המילון למטריצה X
end

%% Signal-to-Noise Ratio (SNR) Calculation for Learned Kernels

% Define M values (sizes of training sets)
M_values = [400, 600, 2000];
SNR_values = zeros(length(M_values), 1); % To store SNR results
g_kernels_learned = cell(length(M_values), 1); % Store learned kernels for each M

% Step 1: Compute the original kernels (g_b^s_0(Λ))
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

% Step 2: Loop over M values
for i = 1:length(M_values)
    M = M_values(i);
    fprintf('\nProcessing M = %d training signals...\n', M);

    % Update the number of training signals
    param.nb_train_signals = M;

    % Generate new training signals
    [TrainSignal, ~] = generate_signals(D, param);

    % Learn the dictionary
    [~, ~, g_ker_learned] = Polynomial_Dictionary_Learning4(TrainSignal, param);

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
end

% Step 3: Display final SNR results
fprintf('\nFinal SNR Results:\n');
for i = 1:length(M_values)
    fprintf('M = %d: SNR builtin = %.2f dB\n', M_values(i), SNR_values(i));
end

%% Function to compute the mean SNR
function mean_snr = compute_mean_snr(g_ker_original, g_ker_learned, ~)
    % Vectorized computation of mean SNR
    error_norms = vecnorm(g_ker_original - g_ker_learned, 2, 1); % Compute L2 norms for each subdictionary
    snr_values = -20 * log10(error_norms); % Compute SNR values for each subdictionary
    mean_snr = mean(snr_values); % Average the SNR values
end
