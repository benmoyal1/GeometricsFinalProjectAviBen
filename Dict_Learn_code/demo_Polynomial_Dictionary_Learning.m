%==========================================================================
     %% Example
%==========================================================================

% Description: Run file that applies the polynomial dictionary learning algorithm
% in the data contained in testdata.mat. The mat file contains the necessary data that are needed 
% to reproduce the synthetic results of Section V.A.1 of the reference paper:

% D. Thanou, D. I Shuman, and P. Frossard, ?Learning Parametric Dictionaries for Signals on Graphs?, 
% Submitted to IEEE Transactions on Signal Processing,
% Available at:  http://arxiv.org/pdf/1401.0887.pdf

clear all
close all

%% Adding the paths
addpath('C:\Users\Cristina\Documents\GitHub\OrganizedFiles\Optimizers'); %Folder conatining the yalmip tools
addpath('C:\Users\Cristina\Documents\GitHub\OrganizedFiles\Pol_Dict_Learn_code\DataSets'); %Folder containing the Data sets
addpath('C:\Users\Cristina\Documents\GitHub\OrganizedFiles\DataSets\Comparison_datasets\'); %Folder containing the copmarison datasets
addpath('C:\Users\Cristina\Documents\GitHub\OrganizedFiles\DataSets\'); %Folder containing the training and verification dataset
path = 'C:\Users\Cristina\Documents\GitHub\OrganizedFiles\Pol_Dict_Learn_code\Results\'; %Folder containing the results to save

%% Loaging the required dataset
flag = 2;
switch flag
    case 1
        load ComparisonDorina.mat
        ds = 'Dataset used: Synthetic data from Dorina';
        load DataSetDorina.mat
    case 2
        load ComparisonLF.mat
        ds = 'Dataset used: data from Cristina';
        load DataSetLF.mat
    case 3
        load ComparisonUber.mat
        ds = 'Dataset used: data from Uber';
        load DataSetUber.mat        
end

%% Set the parameters

switch flag
    case 1 %Dorina
        param.S = 4;  % number of subdictionaries 
        param.epsilon = 0.02; % we assume that epsilon_1 = epsilon_2 = epsilon
        degree = 20;
    case 2 %Cristina
        param.S = 2;  % number of subdictionaries 
        param.epsilon = 0.02; % we assume that epsilon_1 = epsilon_2 = epsilon
        degree = 15;
    case 3 %Uber
        param.S = 2;  % number of subdictionaries
        param.epsilon = 0.02; % we assume that epsilon_1 = epsilon_2 = epsilon
        degree = 15;
end

param.N = 100; % number of nodes in the graph
param.J = param.N * param.S; % total number of atoms 
param.K = degree*ones(1,param.S);
param.T0 = 4; % sparsity level in the training phase
param.c = 1; % spectral control parameters
param.mu = 1e-2; % polynomial regularizer paremeter

%% Plot the random graph

% figure()   
% gplot(A,[XCoords YCoords])

%% Compute the Laplacian and the normalized laplacian operator
    
L = diag(sum(W,2)) - W; % combinatorial Laplacian
param.Laplacian = (diag(sum(W,2)))^(-1/2)*L*(diag(sum(W,2)))^(-1/2); % normalized Laplacian
[param.eigenMat, param.eigenVal] = eig(param.Laplacian); % eigendecomposition of the normalized Laplacian
[param.lambda_sym,index_sym] = sort(diag(param.eigenVal)); % sort the eigenvalues of the normalized Laplacian in descending order

if flag == 1
    comp_eigenVal = param.eigenVal;
end
%% Compute the powers of the Laplacian

for k=0 : max(param.K)
    param.Laplacian_powers{k + 1} = param.Laplacian^k;
end
    
for j=1:param.N
    for i=0:max(param.K)
        param.lambda_powers{j}(i + 1) = param.lambda_sym(j)^(i);
        param.lambda_power_matrix(j,i + 1) = param.lambda_sym(j)^(i);
     end
end
    
%% Polynomial dictionary learning algorithm

param.InitializationMethod =  'Random_kernels';
param.displayProgress = 1;
param.numIteration = 8;
param.plot_kernels = 1; % plot thelearned polynomial kernels after each iteration
param.quadratic = 0; % solve the quadratic program using interior point methods

disp('Starting to train the dictionary');

[Dictionary_Pol,output_Pol,g_ker]  = Polynomial_Dictionary_Learning(TrainSignal, param);

CoefMatrix_Pol = OMP_non_normalized_atoms(Dictionary_Pol,TestSignal, param.T0);
errorTesting_Pol = sqrt(norm(TestSignal - Dictionary_Pol*CoefMatrix_Pol,'fro')^2/size(TestSignal,2));
disp(['The total representation error of the testing signals is: ',num2str(errorTesting_Pol)]);

%% Compute the l-2 norms

lambda_norm = 'is 0 since here we are learning only the kernels'; %norm(comp_eigenVal - eigenVal);
alpha_norm = norm(comp_alpha - output_Pol.alpha);
X_norm = norm(comp_X - CoefMatrix_Pol);
D_norm = norm(comp_D - Dictionary_Pol);
W_norm = 'is 0 since here we are learning only the kernels';

%% Compute the average CPU_time

avgCPU = mean(output_Pol.cpuTime);

%% Save the results to file

% The norms
filename = [path,'Norms.mat'];
save(filename,'lambda_norm','alpha_norm','X_norm','D_norm','W_norm');

% The Output data
filename = [path,'Output.mat'];
learned_alpha = output_Pol.alpha;
save(filename,'ds','Dictionary_Pol','learned_alpha','CoefMatrix_Pol','errorTesting_Pol','avgCPU');

% The kernels plot
figure('Name','Final Kernels')
hold on
for s = 1 : param.S
    plot(param.lambda_sym,g_ker(:,s));
end
hold off

filename = [path,'FinalKernels_plot.png'];
saveas(gcf,filename);

% The CPU time plot
xq = 0:0.2:8;
figure('Name','CPU time per iteration')
vq2 = interp1(1:8,output_Pol.cpuTime,xq,'spline');
plot(1:8,output_Pol.cpuTime,'o',xq,vq2,':.');
xlim([0 8]);

filename = [path,'AvgCPUtime_plot.png'];
saveas(gcf,filename);

