function [Ks, t1s] = X2Ks_large(X)
% 
% Compute Embedding from Large Kernel
% 
% The memory is O(n^2)
% The time is O(n^2 k)
% 
% Output
%         Hs, cell(1, 12)
%         t1s, time for kernel computation
%         t2s, time for embedding
% 

nKernel = 12;
nSmp = size(X,1);
Ks = zeros(nSmp, nSmp, nKernel);
t1s = zeros(1, nKernel);
%*********************************************
% Linear Kernel
%*********************************************
t1_s = tic;
K = full(X * X');
K = kcenter(K);
K = knorm(K);
K = (K + K')/2;
Ks(:, :, 1) = K;
t1s(1) = toc(t1_s);



%*********************************************
% PolyPlus Kernel
%*********************************************
t1_s = tic;
K = full(X * X');
K = (K + 1).^2;
K = kcenter(K);
K = knorm(K);
K = (K + K')/2;
Ks(:, :, 2) = K;
t1s(2) = toc(t1_s);


t1_s = tic;
K = full(X * X');
K = (K + 1).^4;
K = kcenter(K);
K = knorm(K);
K = (K + K')/2;
Ks(:, :, 3) = K;
t1s(3) = toc(t1_s);


%*********************************************
% Polynomial Kernel
%*********************************************
t1_s = tic;
K = full(X * X');
K = K.^2;
K = kcenter(K);
K = knorm(K);
K = (K + K')/2;
Ks(:, :, 4) = K;
t1s(4) = toc(t1_s);

t1_s = tic;
K = full(X * X');
K = K.^4;
K = kcenter(K);
K = knorm(K);
K = (K + K')/2;
Ks(:, :, 5) = K;
t1s(5) = toc(t1_s);


%*********************************************
% Gaussian Kernel
%*********************************************

% Gaussian 2.^-3
t1_s = tic;
D = EuDist2(X, [], 0);
s = mean(mean(D));
K = exp(-D / (2 * 2.^-3 * s) );
K = kcenter(K);
K = knorm(K);
K = (K + K')/2;
Ks(:, :, 6) = K;
t1s(6) = toc(t1_s);


% Gaussian 2.^-2
t1_s = tic;
% D = EuDist2(X, [], 0);
% s = mean(mean(D));
K = exp(-D / (2 * 2.^-2 * s) );
K = kcenter(K);
K = knorm(K);
K = (K + K')/2;
Ks(:, :, 7) = K;
t1s(7) = toc(t1_s);


% Gaussian 2.^-1
t1_s = tic;
% D = EuDist2(X, [], 0);
% s = mean(mean(D));
K = exp(-D / (2 * 2.^-1 * s) );
K = kcenter(K);
K = knorm(K);
K = (K + K')/2;
Ks(:, :, 8) = K;
t1s(8) = toc(t1_s);


% Gaussian 2.^0
t1_s = tic;
% D = EuDist2(X, [], 0);
% s = mean(mean(D));
K = exp(-D / (2 * 2.^0 * s) );
K = kcenter(K);
K = knorm(K);
K = (K + K')/2;
Ks(:, :, 9) = K;
t1s(9) = toc(t1_s);


% Gaussian 2.^1
t1_s = tic;
% D = EuDist2(X, [], 0);
% s = mean(mean(D));
K = exp(-D / (2 * 2.^1 * s) );
K = kcenter(K);
K = knorm(K);
K = (K + K')/2;
Ks(:, :, 10) = K;
t1s(10) = toc(t1_s);


% Gaussian 2.^2
t1_s = tic;
D = EuDist2(X, [], 0);
s = mean(mean(D));
K = exp(-D / (2 * 2.^2 * s) );
K = kcenter(K);
K = knorm(K);
K = (K + K')/2;
Ks(:, :, 11) = K;
t1s(11) = toc(t1_s);


% Gaussian 2.^3
t1_s = tic;
D = EuDist2(X, [], 0);
s = mean(mean(D));
K = exp(-D / (2 * 2.^3 * s) );
K = kcenter(K);
K = knorm(K);
K = (K + K')/2;
Ks(:, :, 12) = K;
t1s(12) = toc(t1_s);
end