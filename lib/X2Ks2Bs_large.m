function [Bs, t1s, t2s] = X2Ks2Bs_large(X, nAnchor, k)
%
% Compute Embedding from Large Kernel
%
% The memory is O(n^2)
% The time is O(n^2 k)
%
% Output
%         Bs, cell(1, 12)
%         t1s, time for kernel computation
%         t2s, time for embedding
%
[nSmp, nFea] = size(X);

nKernel = 12;
t1s = zeros(1, nKernel);
t2s = zeros(1, nKernel);
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
K = K - 10^8*eye(nSmp);
[~, Idx] = sort(K, 1, 'descend');
Idx = Idx(1:k, :);
Idx2 = Idx';
colIdx = Idx2(:);
rowIdx = repmat((1:nSmp)', k, 1);
lidx = sub2ind([nSmp, nSmp], rowIdx, colIdx);
val = K(lidx);
G = sparse(rowIdx, colIdx, val, nSmp, nSmp,nSmp * k);
Ssym = (G + G')/2;
DSsym = 1./sqrt(max(sum(Ssym, 2), eps));
Gnorm = (DSsym * DSsym') .* Ssym;
Gnorm = (Gnorm + Gnorm')/2;
anchor_idx = AnchorSelection_ALG(Gnorm, nAnchor);
rowIdx = repmat((1:nSmp)', k, 1);
Kb = K(anchor_idx, :);
[Kb2, Idx] = sort(Kb, 1, 'descend');
Idx = Idx(1:k, :)';
colIdx = Idx(:);
Kb2 = Kb2(1:k, :)';
val = Kb2(:);
B = sparse(rowIdx, colIdx, val, nSmp, nAnchor, nSmp * k);
B = bsxfun(@rdivide, B, max(sum(B, 2), eps));
B = bsxfun(@times, B, max(sum(B, 1), eps).^(-.5));
Bs{1} = B;
t2s(1) = toc(t2_s);


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
K = K - 10^8*eye(nSmp);
[~, Idx] = sort(K, 1, 'descend');
Idx = Idx(1:k, :);
Idx2 = Idx';
colIdx = Idx2(:);
rowIdx = repmat((1:nSmp)', k, 1);
lidx = sub2ind([nSmp, nSmp], rowIdx, colIdx);
val = K(lidx);
G = sparse(rowIdx, colIdx, val, nSmp, nSmp,nSmp * k);
Ssym = (G + G')/2;
DSsym = 1./sqrt(max(sum(Ssym, 2), eps));
Gnorm = (DSsym * DSsym') .* Ssym;
Gnorm = (Gnorm + Gnorm')/2;
anchor_idx = AnchorSelection_ALG(Gnorm, nAnchor);
rowIdx = repmat((1:nSmp)', k, 1);
Kb = K(anchor_idx, :);
[Kb2, Idx] = sort(Kb, 1, 'descend');
Idx = Idx(1:k, :)';
colIdx = Idx(:);
Kb2 = Kb2(1:k, :)';
val = Kb2(:);
B = sparse(rowIdx, colIdx, val, nSmp, nAnchor, nSmp * k);
B = bsxfun(@rdivide, B, max(sum(B, 2), eps));
B = bsxfun(@times, B, max(sum(B, 1), eps).^(-.5));
Bs{2} = B;
t2s(2) = toc(t2_s);

t1_s = tic;
K = full(X * X');
K = (K + 1).^4;
K = kcenter(K);
K = knorm(K);
K = (K + K')/2;
t1s(3) = toc(t1_s);

t2_s = tic;
K = K - 10^8*eye(nSmp);
[~, Idx] = sort(K, 1, 'descend');
Idx = Idx(1:k, :);
Idx2 = Idx';
colIdx = Idx2(:);
rowIdx = repmat((1:nSmp)', k, 1);
lidx = sub2ind([nSmp, nSmp], rowIdx, colIdx);
val = K(lidx);
G = sparse(rowIdx, colIdx, val, nSmp, nSmp,nSmp * k);
Ssym = (G + G')/2;
DSsym = 1./sqrt(max(sum(Ssym, 2), eps));
Gnorm = (DSsym * DSsym') .* Ssym;
Gnorm = (Gnorm + Gnorm')/2;
anchor_idx = AnchorSelection_ALG(Gnorm, nAnchor);
rowIdx = repmat((1:nSmp)', k, 1);
Kb = K(anchor_idx, :);
[Kb2, Idx] = sort(Kb, 1, 'descend');
Idx = Idx(1:k, :)';
colIdx = Idx(:);
Kb2 = Kb2(1:k, :)';
val = Kb2(:);
B = sparse(rowIdx, colIdx, val, nSmp, nAnchor, nSmp * k);
B = bsxfun(@rdivide, B, max(sum(B, 2), eps));
B = bsxfun(@times, B, max(sum(B, 1), eps).^(-.5));
Bs{3} = B;
t2s(3) = toc(t2_s);

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
K = K - 10^8*eye(nSmp);
[~, Idx] = sort(K, 1, 'descend');
Idx = Idx(1:k, :);
Idx2 = Idx';
colIdx = Idx2(:);
rowIdx = repmat((1:nSmp)', k, 1);
lidx = sub2ind([nSmp, nSmp], rowIdx, colIdx);
val = K(lidx);
G = sparse(rowIdx, colIdx, val, nSmp, nSmp,nSmp * k);
Ssym = (G + G')/2;
DSsym = 1./sqrt(max(sum(Ssym, 2), eps));
Gnorm = (DSsym * DSsym') .* Ssym;
Gnorm = (Gnorm + Gnorm')/2;
anchor_idx = AnchorSelection_ALG(Gnorm, nAnchor);
rowIdx = repmat((1:nSmp)', k, 1);
Kb = K(anchor_idx, :);
[Kb2, Idx] = sort(Kb, 1, 'descend');
Idx = Idx(1:k, :)';
colIdx = Idx(:);
Kb2 = Kb2(1:k, :)';
val = Kb2(:);
B = sparse(rowIdx, colIdx, val, nSmp, nAnchor, nSmp * k);
B = bsxfun(@rdivide, B, max(sum(B, 2), eps));
B = bsxfun(@times, B, max(sum(B, 1), eps).^(-.5));
Bs{4} = B;
t2s(4) = toc(t2_s);

t1_s = tic;
K = full(X * X');
K = K.^4;
K = kcenter(K);
K = knorm(K);
K = (K + K')/2;
t1s(5) = toc(t1_s);

t2_s = tic;
K = K - 10^8*eye(nSmp);
[~, Idx] = sort(K, 1, 'descend');
Idx = Idx(1:k, :);
Idx2 = Idx';
colIdx = Idx2(:);
rowIdx = repmat((1:nSmp)', k, 1);
lidx = sub2ind([nSmp, nSmp], rowIdx, colIdx);
val = K(lidx);
G = sparse(rowIdx, colIdx, val, nSmp, nSmp,nSmp * k);
Ssym = (G + G')/2;
DSsym = 1./sqrt(max(sum(Ssym, 2), eps));
Gnorm = (DSsym * DSsym') .* Ssym;
Gnorm = (Gnorm + Gnorm')/2;
anchor_idx = AnchorSelection_ALG(Gnorm, nAnchor);
rowIdx = repmat((1:nSmp)', k, 1);
Kb = K(anchor_idx, :);
[Kb2, Idx] = sort(Kb, 1, 'descend');
Idx = Idx(1:k, :)';
colIdx = Idx(:);
Kb2 = Kb2(1:k, :)';
val = Kb2(:);
B = sparse(rowIdx, colIdx, val, nSmp, nAnchor, nSmp * k);
B = bsxfun(@rdivide, B, max(sum(B, 2), eps));
B = bsxfun(@times, B, max(sum(B, 1), eps).^(-.5));
Bs{5} = B;
t2s(5) = toc(t2_s);

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
K = K - 10^8*eye(nSmp);
[~, Idx] = sort(K, 1, 'descend');
Idx = Idx(1:k, :);
Idx2 = Idx';
colIdx = Idx2(:);
rowIdx = repmat((1:nSmp)', k, 1);
lidx = sub2ind([nSmp, nSmp], rowIdx, colIdx);
val = K(lidx);
G = sparse(rowIdx, colIdx, val, nSmp, nSmp,nSmp * k);
Ssym = (G + G')/2;
DSsym = 1./sqrt(max(sum(Ssym, 2), eps));
Gnorm = (DSsym * DSsym') .* Ssym;
Gnorm = (Gnorm + Gnorm')/2;
anchor_idx = AnchorSelection_ALG(Gnorm, nAnchor);
rowIdx = repmat((1:nSmp)', k, 1);
Kb = K(anchor_idx, :);
[Kb2, Idx] = sort(Kb, 1, 'descend');
Idx = Idx(1:k, :)';
colIdx = Idx(:);
Kb2 = Kb2(1:k, :)';
val = Kb2(:);
B = sparse(rowIdx, colIdx, val, nSmp, nAnchor, nSmp * k);
B = bsxfun(@rdivide, B, max(sum(B, 2), eps));
B = bsxfun(@times, B, max(sum(B, 1), eps).^(-.5));
Bs{6} = B;
t2s(6) = toc(t2_s);

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
K = K - 10^8*eye(nSmp);
[~, Idx] = sort(K, 1, 'descend');
Idx = Idx(1:k, :);
Idx2 = Idx';
colIdx = Idx2(:);
rowIdx = repmat((1:nSmp)', k, 1);
lidx = sub2ind([nSmp, nSmp], rowIdx, colIdx);
val = K(lidx);
G = sparse(rowIdx, colIdx, val, nSmp, nSmp,nSmp * k);
Ssym = (G + G')/2;
DSsym = 1./sqrt(max(sum(Ssym, 2), eps));
Gnorm = (DSsym * DSsym') .* Ssym;
Gnorm = (Gnorm + Gnorm')/2;
anchor_idx = AnchorSelection_ALG(Gnorm, nAnchor);
rowIdx = repmat((1:nSmp)', k, 1);
Kb = K(anchor_idx, :);
[Kb2, Idx] = sort(Kb, 1, 'descend');
Idx = Idx(1:k, :)';
colIdx = Idx(:);
Kb2 = Kb2(1:k, :)';
val = Kb2(:);
B = sparse(rowIdx, colIdx, val, nSmp, nAnchor, nSmp * k);
B = bsxfun(@rdivide, B, max(sum(B, 2), eps));
B = bsxfun(@times, B, max(sum(B, 1), eps).^(-.5));
Bs{7} = B;
t2s(7) = toc(t2_s);

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
K = K - 10^8*eye(nSmp);
[~, Idx] = sort(K, 1, 'descend');
Idx = Idx(1:k, :);
Idx2 = Idx';
colIdx = Idx2(:);
rowIdx = repmat((1:nSmp)', k, 1);
lidx = sub2ind([nSmp, nSmp], rowIdx, colIdx);
val = K(lidx);
G = sparse(rowIdx, colIdx, val, nSmp, nSmp,nSmp * k);
Ssym = (G + G')/2;
DSsym = 1./sqrt(max(sum(Ssym, 2), eps));
Gnorm = (DSsym * DSsym') .* Ssym;
Gnorm = (Gnorm + Gnorm')/2;
anchor_idx = AnchorSelection_ALG(Gnorm, nAnchor);
rowIdx = repmat((1:nSmp)', k, 1);
Kb = K(anchor_idx, :);
[Kb2, Idx] = sort(Kb, 1, 'descend');
Idx = Idx(1:k, :)';
colIdx = Idx(:);
Kb2 = Kb2(1:k, :)';
val = Kb2(:);
B = sparse(rowIdx, colIdx, val, nSmp, nAnchor, nSmp * k);
B = bsxfun(@rdivide, B, max(sum(B, 2), eps));
B = bsxfun(@times, B, max(sum(B, 1), eps).^(-.5));
Bs{8} = B;
t2s(8) = toc(t2_s);


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
K = K - 10^8*eye(nSmp);
[~, Idx] = sort(K, 1, 'descend');
Idx = Idx(1:k, :);
Idx2 = Idx';
colIdx = Idx2(:);
rowIdx = repmat((1:nSmp)', k, 1);
lidx = sub2ind([nSmp, nSmp], rowIdx, colIdx);
val = K(lidx);
G = sparse(rowIdx, colIdx, val, nSmp, nSmp,nSmp * k);
Ssym = (G + G')/2;
DSsym = 1./sqrt(max(sum(Ssym, 2), eps));
Gnorm = (DSsym * DSsym') .* Ssym;
Gnorm = (Gnorm + Gnorm')/2;
anchor_idx = AnchorSelection_ALG(Gnorm, nAnchor);
rowIdx = repmat((1:nSmp)', k, 1);
Kb = K(anchor_idx, :);
[Kb2, Idx] = sort(Kb, 1, 'descend');
Idx = Idx(1:k, :)';
colIdx = Idx(:);
Kb2 = Kb2(1:k, :)';
val = Kb2(:);
B = sparse(rowIdx, colIdx, val, nSmp, nAnchor, nSmp * k);
B = bsxfun(@rdivide, B, max(sum(B, 2), eps));
B = bsxfun(@times, B, max(sum(B, 1), eps).^(-.5));
Bs{9} = B;
t2s(9) = toc(t2_s);

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
K = K - 10^8*eye(nSmp);
[~, Idx] = sort(K, 1, 'descend');
Idx = Idx(1:k, :);
Idx2 = Idx';
colIdx = Idx2(:);
rowIdx = repmat((1:nSmp)', k, 1);
lidx = sub2ind([nSmp, nSmp], rowIdx, colIdx);
val = K(lidx);
G = sparse(rowIdx, colIdx, val, nSmp, nSmp,nSmp * k);
Ssym = (G + G')/2;
DSsym = 1./sqrt(max(sum(Ssym, 2), eps));
Gnorm = (DSsym * DSsym') .* Ssym;
Gnorm = (Gnorm + Gnorm')/2;
anchor_idx = AnchorSelection_ALG(Gnorm, nAnchor);
rowIdx = repmat((1:nSmp)', k, 1);
Kb = K(anchor_idx, :);
[Kb2, Idx] = sort(Kb, 1, 'descend');
Idx = Idx(1:k, :)';
colIdx = Idx(:);
Kb2 = Kb2(1:k, :)';
val = Kb2(:);
B = sparse(rowIdx, colIdx, val, nSmp, nAnchor, nSmp * k);
B = bsxfun(@rdivide, B, max(sum(B, 2), eps));
B = bsxfun(@times, B, max(sum(B, 1), eps).^(-.5));
Bs{10} = B;
t2s(10) = toc(t2_s);

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
K = K - 10^8*eye(nSmp);
[~, Idx] = sort(K, 1, 'descend');
Idx = Idx(1:k, :);
Idx2 = Idx';
colIdx = Idx2(:);
rowIdx = repmat((1:nSmp)', k, 1);
lidx = sub2ind([nSmp, nSmp], rowIdx, colIdx);
val = K(lidx);
G = sparse(rowIdx, colIdx, val, nSmp, nSmp,nSmp * k);
Ssym = (G + G')/2;
DSsym = 1./sqrt(max(sum(Ssym, 2), eps));
Gnorm = (DSsym * DSsym') .* Ssym;
Gnorm = (Gnorm + Gnorm')/2;
anchor_idx = AnchorSelection_ALG(Gnorm, nAnchor);
rowIdx = repmat((1:nSmp)', k, 1);
Kb = K(anchor_idx, :);
[Kb2, Idx] = sort(Kb, 1, 'descend');
Idx = Idx(1:k, :)';
colIdx = Idx(:);
Kb2 = Kb2(1:k, :)';
val = Kb2(:);
B = sparse(rowIdx, colIdx, val, nSmp, nAnchor, nSmp * k);
B = bsxfun(@rdivide, B, max(sum(B, 2), eps));
B = bsxfun(@times, B, max(sum(B, 1), eps).^(-.5));
Bs{11} = B;
t2s(11) = toc(t2_s);

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
K = K - 10^8*eye(nSmp);
[~, Idx] = sort(K, 1, 'descend');
Idx = Idx(1:k, :);
Idx2 = Idx';
colIdx = Idx2(:);
rowIdx = repmat((1:nSmp)', k, 1);
lidx = sub2ind([nSmp, nSmp], rowIdx, colIdx);
val = K(lidx);
G = sparse(rowIdx, colIdx, val, nSmp, nSmp,nSmp * k);
Ssym = (G + G')/2;
DSsym = 1./sqrt(max(sum(Ssym, 2), eps));
Gnorm = (DSsym * DSsym') .* Ssym;
Gnorm = (Gnorm + Gnorm')/2;
anchor_idx = AnchorSelection_ALG(Gnorm, nAnchor);
rowIdx = repmat((1:nSmp)', k, 1);
Kb = K(anchor_idx, :);
[Kb2, Idx] = sort(Kb, 1, 'descend');
Idx = Idx(1:k, :)';
colIdx = Idx(:);
Kb2 = Kb2(1:k, :)';
val = Kb2(:);
B = sparse(rowIdx, colIdx, val, nSmp, nAnchor, nSmp * k);
B = bsxfun(@rdivide, B, max(sum(B, 2), eps));
B = bsxfun(@times, B, max(sum(B, 1), eps).^(-.5));
Bs{12} = B;
t2s(12) = toc(t2_s);
end