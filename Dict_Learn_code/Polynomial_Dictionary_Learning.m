function [recovered_Dictionary,output,g_ker] = Polynomial_Dictionary_Learning(Y, param)

% =========================================================================
%                  Polynomial Dictionary Learning Algorithm
% =========================================================================
% Description: The function implements the polynomial dictionary learning algorithm 
%               presented in: 
%               "Learning Parametric Dictionaries for Signals on Graphs", 
%                by D. Thanou, D. I Shuman, and P. Frossard
%                arXiv: 1401.0887, January, 2014. 
%% Input: 
%      Y:                      Set of training signals
%      param:                     Structure that includes all required parameters for the
%                                 algorithm. It contains the following fields:
%      param.N:                   number of nodes in the graph
%      param.S:                   number of subdictionaries
%      param.K:                   polynomial degree in each subdictionary
%      param.T0:                  sparsity level in the learning phase
%      param.Laplacian_powers:      a cell that contains the param.K powers of the graph Laplacian 
%      param.lambda_sym:           eigenvalues of the normalized Laplacian
%      param.lambda_powers:        a cell that contains the param.K powers of the eigenvalues of the graph Laplacian 
%      param.numIteration:        (optional) maximum number of iterations. When it is not
%                                 provided, it is set to 100 
%      param.InitializationMethod: (optional) A method for intializing the dictionary. 
%                                  If
%                                  param.InitializationMethod ='Random_kernels'(default),initialize using the function initialize_dictionary(),
%                                  else if 
%                                  param.InitializationMethod ='GivenMatrix'use a initial given dictionary (e.g., Spectral Graph Wavelet Dictionary)                                 
%      param.quadratic:           (optional) if 0 (by default), it uses interior point methods to
%                                 solve the dictionary ipdate step, otherwise uses ADMM.
%      param.plot_kernels:        (optional) if 1 (by default), it plots the learned kernel after each iteration   
%      param.displayProgress:     (optional) if 1 (by default), it prints the total mean square representation error.
%      
%                                  
%

%% Output:
%     recoveredDictionary: The recovered polynomial dictionary
%     output: structure that includes all the following fields
%     output.alpha: learned polynomial coefficients
%     output.CoefMatrix: Sparse codes of the training signals

% =========================================================================

%%-----------------------------------------------
%%-------------- Set parameters------------------
%%-----------------------------------------------
lambda_sym = param.lambda_sym;
lambda_powers = param.lambda_powers;
Laplacian_powers = param.Laplacian_powers;

if (~isfield(param,'displayProgress'))
    param.displayProgress = 1;
end

if (~isfield(param,'quadratic'))
    param.quadratic = 0;
end

if (~isfield(param,'plot_kernels'))
    param.plot_kernels = 1;
end

if (~isfield(param,'numIteration'))
    param.numIteration = 100;
end

if (~isfield(param,'InitializationMethod'))
    param.InitializationMethod = 'Random_kernels';
end

color_matrix = ['b', 'r', 'g', 'c', 'm', 'k', 'y'];
 
%%-----------------------------------------------
%% Initializing the dictionary
%%-----------------------------------------------

if (strcmp(param.InitializationMethod,'Random_kernels')) 
    [Dictionary(:,1 : param.J)] = initialize_dictionary(param);
   
    
elseif (strcmp(param.InitializationMethod,'GivenMatrix'))
        Dictionary(:,1 : param.J) = param.initialDictionary(:,1 : param.J);  %initialize with a given initialization dictionary
else 
    disp('Initialization method is not valid')
end


%%----------------------------------------------------
%%  Graph Dictionary Learning Algorithm
%%----------------------------------------------------

cpuTime = zeros(1,param.numIteration);
for iterNum = 1 : param.numIteration
%%----------------------------------------------------
   %%  Sparse Coding Step (OMP)
%%----------------------------------------------------
      CoefMatrix = OMP_non_normalized_atoms(Dictionary,Y, param.T0);
    
%%----------------------------------------------------
   %%  Dictionary Update Step
%%---------------------------------------------------
          
if (param.quadratic == 0)
   if (iterNum == 1)
    disp('solving the quadratic problem with YALMIP...')
   end
    [alpha, diagnostics] = coefficient_update_interior_point(Y,CoefMatrix,param,'sdpt3');
    cpuTime(iterNum) = diagnostics.solveroutput.info.cputime;
else
   if (iterNum == 1)
    disp('solving the quadratic problem with ADMM...')
   end
    [Q1,Q2, B, h] = compute_ADMM_entries(Y, param, Laplacian_powers, CoefMatrix);
     alpha = coefficient_upadate_ADMM(Q1, Q2, B, h);
end

g_ker = zeros(param.N, param.S);
r = 0;
for i = 1 : param.S
    for n = 1 : param.N
    p = 0;
    for l = 0 : param.K(i)
        p = p +  alpha(l + 1 + r)*lambda_powers{n}(l + 1);
    end
    g_ker(n,i) = p;
    end
    r = sum(param.K(1:i)) + i;
end

if (param.plot_kernels == 1) 
  figure()
  hold on
  for s = 1 : param.S
      plot(lambda_sym,g_ker(:,s),num2str(color_matrix(s)));
  end
  hold off
end



r = 0;
for j = 1 : param.S
    D = zeros(param.N);
    for ii = 0 : param.K(j)
        D = D +  alpha(ii + 1 + r) * Laplacian_powers{ii + 1};
    end
    r = sum(param.K(1:j)) + j;
    Dictionary(:,1 + (j - 1) * param.N : j * param.N) = D;
end



%%---------------------------------------------------
%% Plot the progress and save average computation time
%%--------------------------------------------------

if (iterNum>1 && param.displayProgress==1)
     output.totalError(iterNum - 1) = sqrt(sum(sum((Y-Dictionary * CoefMatrix).^2))/numel(Y));
     disp(['Iteration   ',num2str(iterNum),'   Total error is: ',num2str(output.totalError(iterNum-1))]);
end
    

end

output.cpuTime = cpuTime;
output.CoefMatrix = CoefMatrix;
output.alpha =  alpha;
recovered_Dictionary = Dictionary;






function Initial_Dictionary = initialize_dictionary(param)

%======================================================
   %%  Dictionary Initialization
%======================================================


%% Input:
%         param.N:        number of nodes of the graph
%         param.J:        number of atoms in the dictionary 
%         param.S:        number of subdictionaries
%         param.eigenMat: eigenvectors of the graph Laplacian
%         param.c:        upper-bound on the spectral representation of the kernels 
%           
%% Output: 
%         Initial_Dictionary: A matrix for initializing the dictionary
%======================================================


J = param.J;
N = param.N;
S = param.S;
c = param.c;
Initial_Dictionary = zeros(N,J);

for i = 1 : S
   
    tmpLambda = c * rand(param.N);
       
    if isempty(tmpLambda)
        disp('Initialization fails');
        exit;
    end
    
    tmpLambda = diag(tmpLambda(randperm(N)));
    Initial_Dictionary(:,1 + (i - 1) * N : i * N) = param.eigenMat * tmpLambda * param.eigenMat';
end



function [Q1,Q2, B, h] = compute_ADMM_entries(Y, param, Laplacian_powers, X)
%======================================================
   %% Compute the required entries for ADMM
%======================================================
% Description: Find Q1, Q2, B, h such that the quadratic program is
% expressed as: 
%       minimize     (1/2)*alpha'*Q1*alpha-Q2*alpha
%       subject to   B*alpha<=h
%======================================================

N = param.N;
c = param.c;
epsilon = param.epsilon;
mu = param.mu;
S = param.S;
K = max(param.K);
Lambda = param.lambda_power_matrix;



B1 = sparse(kron(eye(S),Lambda));
B2 = kron(ones(1,S),Lambda);

Phi = zeros(S*(K+1),1);
for i = 1 : N
         r = 0;
        for s = 1 : S
            for k = 0 : K
                Phi(k + 1 + r,(i - 1)*size(Y,2) +1 : i*size(Y,2)) = Laplacian_powers{k + 1}(i,:)*X((s-1)*N + 1 : s*N,1 : end);
            end
            r = sum(param.K(1 : s)) + s;
        end
end

YPhi = (Phi*vec(Y'))';
PhiPhiT = Phi*Phi';

Q2 = YPhi;
Q1 = PhiPhiT + mu*eye(size(PhiPhiT,2));


B = [B1; -B1; B2; -B2];
h = [c*ones(size(B1,1),1);zeros(size(B1,1),1);(c + epsilon)*ones(size(B2,1),1); -(c - epsilon)*ones(size(B2,1),1)];







