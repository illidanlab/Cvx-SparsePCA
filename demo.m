% Convex Sparse PCA
% z_SVRG: Sparse load corresponding to the first principal component
% eta: Step size
% m: Number of iterations in each epoch
% rho1: Regularization parameter 

% Inci M. Baytas, Computer Science Department, MSU
% April, 2016
X = rand(500,50);

[~,dim] = size(X);
z = randn(dim, 1);
w = randn(dim, 1);

X = bsxfun(@rdivide, X, (sum(X.^2,2)).^0.5);

Cov_X = X' * X;

[~,lambda_1] = eigs(Cov_X,1);
lambda = 2*lambda_1;
sigma = lambda - lambda_1;
eta = 0.1/lambda_1;

m = 500;

rho1 = 0.3;
maxEpoch = 50;

[z_SVRG,~] = Prox_SVRG_convex(X', z, rho1, maxEpoch,m,eta,lambda,w,Cov_X); 
