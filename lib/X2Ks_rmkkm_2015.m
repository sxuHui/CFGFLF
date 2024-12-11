function [Ks, t1s] = X2Ks_rmkkm_2015(X)
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
% The code of RMKKM [IJCAI 2015] is
%         PolynomialDegrees = [2, 4];
%         PolyPlusDegrees = [2, 4];
%         GaussianDegrees = [0.01, 0.05, 0.1, 1, 10, 50, 100];
% The paper of RMKKM [IJCAI 2015] is
%         PolynomialDegrees = [0, 1];
%         PolyPlusDegrees = [2, 4];
%         GaussianDegrees = [0.01, 0.05, 0.1, 1, 10, 50, 100];



nKernel = 12;
nSmp = size(X,1);
Ks = zeros(nSmp, nSmp, nKernel);
t1s = zeros(1, nKernel);
%*********************************************
% Linear Kernel
%*********************************************
t1_s = tic;
K0 = full(X * X');
% K = kcenter(K0);
K = K0;
K = knorm(K);
K = (K + K')/2;
Ks(:, :, 1) = K;
t1s(1) = toc(t1_s);



%*********************************************
% PolyPlus Kernel
%*********************************************
t1_s = tic;
% K = full(X * X');
K = (K0 + 0).^2;
% K = kcenter(K);
K = knorm(K);
K = (K + K')/2;
Ks(:, :, 2) = K;
t1s(2) = toc(t1_s);


t1_s = tic;
% K = full(X * X');
K = (K0 + 0).^4;
% K = kcenter(K);
K = knorm(K);
K = (K + K')/2;
Ks(:, :, 3) = K;
t1s(3) = toc(t1_s);


%*********************************************
% Polynomial Kernel
%*********************************************
t1_s = tic;
% K = full(X * X');
K = (K0 + 1).^2;
% K = kcenter(K);
K = knorm(K);
K = (K + K')/2;
Ks(:, :, 4) = K;
t1s(4) = toc(t1_s);

t1_s = tic;
% K = full(X * X');
K = (K0 + 1).^4;
% K = kcenter(K);
K = knorm(K);
K = (K + K')/2;
Ks(:, :, 5) = K;
t1s(5) = toc(t1_s);
clear K0;

%*********************************************
% Gaussian Kernel
%*********************************************

% Gaussian 2.^-3
t1_s = tic;
D = EuDist2(X, [], 0);
s = sqrt(max(max(D)));
K = exp(-D / (2 * (0.01 * s)^2 ));
% K = kcenter(K);
K = knorm(K);
K = (K + K')/2;
Ks(:, :, 6) = K;
t1s(6) = toc(t1_s);


% Gaussian 2.^-2
t1_s = tic;
% D = EuDist2(X, [], 0);
% s = sqrt(max(max(D)));
K = exp(-D / (2 * (0.05 * s)^2 ));
% K = kcenter(K);
K = knorm(K);
K = (K + K')/2;
Ks(:, :, 7) = K;
t1s(7) = toc(t1_s);


% Gaussian 2.^-1
t1_s = tic;
% D = EuDist2(X, [], 0);
% s = sqrt(max(max(D)));
K = exp(-D / (2 * (0.1 * s)^2 ));
% K = kcenter(K);
K = knorm(K);
K = (K + K')/2;
Ks(:, :, 8) = K;
t1s(8) = toc(t1_s);


% Gaussian 2.^0
t1_s = tic;
% D = EuDist2(X, [], 0);
% s = sqrt(max(max(D)));
K = exp(-D / (2 * (1 * s)^2 ));
% K = kcenter(K);
K = knorm(K);
K = (K + K')/2;
Ks(:, :, 9) = K;
t1s(9) = toc(t1_s);


% Gaussian 2.^1
t1_s = tic;
% D = EuDist2(X, [], 0);
% s = sqrt(max(max(D)));
K = exp(-D / (2 * (10 * s)^2 ));
% K = kcenter(K);
K = knorm(K);
K = (K + K')/2;
Ks(:, :, 10) = K;
t1s(10) = toc(t1_s);


% Gaussian 2.^2
t1_s = tic;
D = EuDist2(X, [], 0);
s = sqrt(max(max(D)));
K = exp(-D / (2 * (50 * s)^2 ));
% K = kcenter(K);
K = knorm(K);
K = (K + K')/2;
Ks(:, :, 11) = K;
t1s(11) = toc(t1_s);


% Gaussian 2.^3
t1_s = tic;
D = EuDist2(X, [], 0);
s = sqrt(max(max(D)));
K = exp(-D / (2 * (100 * s)^2 ));
% K = kcenter(K);
K = knorm(K);
K = (K + K')/2;
Ks(:, :, 12) = K;
t1s(12) = toc(t1_s);
end