function [Hs, t1s, t2s] = X2Ks2Hs_large(X, nCluster)
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

nSmp = size(X, 1);
nKernel = 12;
t1s = zeros(1, nKernel);
t2s = zeros(1, nKernel);
Hs = zeros(nSmp, nCluster, nKernel);
%*********************************************
% Linear Kernel
%*********************************************
t1_s = tic;
K = full(X * X');
K = kcenter(K);
K = knorm(K);
K = (K + K')/2;
t1s(1) = toc(t1_s);

t2_s = tic;
opt.disp = 0;
[H, ~] = eigs(K, nCluster, 'la', opt);
Hs(:,:,1) = H;
t2s(1) = toc(t2_s);
clear K;

%*********************************************
% PolyPlus Kernel
%*********************************************
t1_s = tic;
K = full(X * X');
K = (K + 1).^2;
K = kcenter(K);
K = knorm(K);
K = (K + K')/2;
t1s(2) = toc(t1_s);

t2_s = tic;
opt.disp = 0;
[H, ~] = eigs(K, nCluster, 'la', opt);
Hs(:,:,2) = H;
t2s(2) = toc(t2_s);
clear K;

t1_s = tic;
K = full(X * X');
K = (K + 1).^4;
K = kcenter(K);
K = knorm(K);
K = (K + K')/2;
t1s(3) = toc(t1_s);

t2_s = tic;
opt.disp = 0;
[H, ~] = eigs(K, nCluster, 'la', opt);
Hs(:,:,3) = H;
t2s(3) = toc(t2_s);
clear K;
%*********************************************
% Polynomial Kernel
%*********************************************
t1_s = tic;
K = full(X * X');
K = K.^2;
K = kcenter(K);
K = knorm(K);
K = (K + K')/2;
t1s(4) = toc(t1_s);

t2_s = tic;
opt.disp = 0;
[H, ~] = eigs(K, nCluster, 'la', opt);
Hs(:,:,4) = H;
t2s(4) = toc(t2_s);
clear K;

t1_s = tic;
K = full(X * X');
K = K.^4;
K = kcenter(K);
K = knorm(K);
K = (K + K')/2;
t1s(5) = toc(t1_s);

t2_s = tic;
opt.disp = 0;
[H, ~] = eigs(K, nCluster, 'la', opt);
Hs(:,:,5) = H;
t2s(5) = toc(t2_s);
clear K;
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
t1s(6) = toc(t1_s);

t2_s = tic;
opt.disp = 0;
[H, ~] = eigs(K, nCluster, 'la', opt);
Hs(:,:,6) = H;
t2s(6) = toc(t2_s);
clear K;

% Gaussian 2.^-2
t1_s = tic;
D = EuDist2(X, [], 0);
s = mean(mean(D));
K = exp(-D / (2 * 2.^-2 * s) );
K = kcenter(K);
K = knorm(K);
K = (K + K')/2;
t1s(7) = toc(t1_s);

t2_s = tic;
opt.disp = 0;
[H, ~] = eigs(K, nCluster, 'la', opt);
Hs(:,:,7) = H;
t2s(7) = toc(t2_s);
clear K;

% Gaussian 2.^-1
t1_s = tic;
D = EuDist2(X, [], 0);
s = mean(mean(D));
K = exp(-D / (2 * 2.^-1 * s) );
K = kcenter(K);
K = knorm(K);
K = (K + K')/2;
t1s(8) = toc(t1_s);

t2_s = tic;
opt.disp = 0;
[H, ~] = eigs(K, nCluster, 'la', opt);
Hs(:,:,8) = H;
t2s(8) = toc(t2_s);
clear K;

% Gaussian 2.^0
t1_s = tic;
D = EuDist2(X, [], 0);
s = mean(mean(D));
K = exp(-D / (2 * 2.^0 * s) );
K = kcenter(K);
K = knorm(K);
K = (K + K')/2;
t1s(9) = toc(t1_s);

t2_s = tic;
opt.disp = 0;
[H, ~] = eigs(K, nCluster, 'la', opt);
Hs(:,:,9) = H;
t2s(9) = toc(t2_s);
clear K;

% Gaussian 2.^1
t1_s = tic;
D = EuDist2(X, [], 0);
s = mean(mean(D));
K = exp(-D / (2 * 2.^1 * s) );
K = kcenter(K);
K = knorm(K);
K = (K + K')/2;
t1s(10) = toc(t1_s);

t2_s = tic;
opt.disp = 0;
[H, ~] = eigs(K, nCluster, 'la', opt);
Hs(:,:,10) = H;
t2s(10) = toc(t2_s);
clear K;

% Gaussian 2.^2
t1_s = tic;
D = EuDist2(X, [], 0);
s = mean(mean(D));
K = exp(-D / (2 * 2.^2 * s) );
K = kcenter(K);
K = knorm(K);
K = (K + K')/2;
t1s(11) = toc(t1_s);

t2_s = tic;
opt.disp = 0;
[H, ~] = eigs(K, nCluster, 'la', opt);
Hs(:,:,11) = H;
t2s(11) = toc(t2_s);
clear K;

% Gaussian 2.^3
t1_s = tic;
D = EuDist2(X, [], 0);
s = mean(mean(D));
K = exp(-D / (2 * 2.^3 * s) );
K = kcenter(K);
K = knorm(K);
K = (K + K')/2;
t1s(12) = toc(t1_s);

t2_s = tic;
opt.disp = 0;
[H, ~] = eigs(K, nCluster, 'la', opt);
Hs(:,:,12) = H;
t2s(12) = toc(t2_s);
clear K;
end