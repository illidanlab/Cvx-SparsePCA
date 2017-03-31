% Proximal Stochastic Gradient Descent with Variance Reduction 
% Inputs:
% X : nxd data matrix
% w_tilde : Initial loading vector for the first principal component
% rho1 : regularization parameter

% Inci M. Baytas, Computer Science Department, MSU
% December, 2015
function [z_tilde, funcVal, iter] = Prox_SVRG_convex(X, z_tilde, rho1, maxEpoch,m,eta,lambda,w,Cov_X)

funcVal = zeros(maxIter+1,1);

[d,n] = size(X);

Cov = (1/n)*Cov_X;

full_cov = (lambda*eye(d) - Cov);

funcVal(1) = 0.5 * z_tilde' * full_cov * z_tilde - w' * z_tilde +rho1 * sum(abs(z_tilde));


for iter = 1:maxEpoch
    
    v_tilde = full_cov * z_tilde - w; % mean gradient at estimated point
    
    zk_1 = z_tilde;
    Sum_z = 0;
    
    for k = 1:m
        
        idx_k = randi(n);
        x = X(:,idx_k);
        
        vk = zk_1 - eta*( (lambda*eye(d)*(zk_1 - z_tilde)) - (x*(x' * (zk_1 - z_tilde))) + v_tilde);                               
        zk = l1_projection(vk,rho1*eta);
        zk_1 = zk;
        Sum_z = Sum_z + zk;
        
    end
    
    z_tilde =Sum_z / m;% wk;
    
 
    funcVal(iter+1) = 0.5 * z_tilde' * full_cov * z_tilde - w' * z_tilde + rho1 * sum(abs(z_tilde));

    if iter>1
        if abs(funcVal(iter)-funcVal(iter-1)) < 10^(-5)
            break;
        end
    end
   

    fprintf('Iter: %d \t cost: %f\n',iter,funcVal(iter)); 
    fprintf('Number of Non-zero: %d\n',nnz(z_tilde));


end



end

function z = l1_projection (v, beta)
    % this projection calculates
    % argmin_z = 0.5 * \|z-v\|_2^2 + beta \|z\|_1
    % z: solution
    % l1_comp_val: value of l1 component (\|z\|_1)
    z = sign(v) .* max(0, abs(v)- beta); 
end


