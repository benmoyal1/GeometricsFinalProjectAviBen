function [alpha]=coefficient_upadate_ADMM(Q1, Q2, B, h)
% =========================================================================
   %%  Update the polynomial coefficients using ADMM
% =========================================================================

% Description: solves the following problem:

%       minimize     (1/2)*alpha'*Q1*alpha-Q2*alpha
%       subject to   B*alpha<=h

% Input: Q1, Q2, B, h
% Output: alpha

% =========================================================================

%%-----------------------------------------------
%%-------------- Set parameters------------------
%%-----------------------------------------------
max_iter = 1000;
epsilon_abs = 1e-4;
epsilon_rel = 1e-3;
n = size(B,2);
p = size(B,1);
m = size(Q1,2);
rho = 0.3; % the augmented Lagrangian parameter
t = 1.5; % the over-relaxation parameter; typical values are between 1.0 and 1.8

%%-----------------------------------------------
%%----------- Initialize variables --------------
%%-----------------------------------------------
alpha = zeros(m,1);
beta = zeros(size(h,1),1);
lambda_d = zeros(size(h,1),1);

%%-----------------------------------------------
%%----------- ADMM ------------------------------
%%-----------------------------------------------

for iter = 1 : max_iter
 
%%-----------------------------------------------
%%---------- Step 1: alpha update ---------------
%%-----------------------------------------------

    if iter > 1
        alpha = R \ (R' \ (Q2' - rho*B'*(beta - h) - B'*lambda_d));
    else
        R = chol(Q1 + rho*(B'*B));
        alpha = R \ (R' \ (Q2' - rho*B'*(beta - h) - B'*lambda_d));
    end
   

%%-----------------------------------------------
%%----- Step 2: beta update with relaxation -----
%%-----------------------------------------------

    beta_old = beta;
    B_alpha_hat = t*B*alpha - (1 - t)*(beta_old - h);
    beta_hat = h - B_alpha_hat - lambda_d/rho;
    beta = max(0,beta_hat);
    
%%-----------------------------------------------
%%---- Step 3: dual update with relaxation ------
%%-----------------------------------------------
   
    lambda_d = lambda_d + rho*(B_alpha_hat + beta - h);
   
%%-----------------------------------------------
%%---- Check the termination criteria -----------
%%-----------------------------------------------

    r_norm = norm(B*alpha + beta - h);
    s_norm = norm(rho*B'*(beta - beta_old));
    
    epsilon_pri = abs(p)*epsilon_abs + epsilon_rel*max(norm(B*alpha), max(norm(beta), norm(h)));
    epsilon_dual = abs(n)*epsilon_abs + epsilon_rel*norm(B'*lambda_d);
    
   
   fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t \n', iter, ...
            r_norm, epsilon_pri, ...
            s_norm, epsilon_dual);

    if (r_norm < epsilon_pri) && (s_norm < epsilon_dual)
        break;
    end
end