function [Iabel, Ws, alpha, beta, objHistory] = CFGFLF_fast(Hs, nCluster, LHs, Y)
nView = length(Hs);
nSmp = size(Hs{1}, 1);


%*************************************************
% Initialization of Ws
%*************************************************
Ws = cell(1, nView);
for iView = 1 : nView
    Ws{iView} = eye(size(Hs{iView}, 2), nCluster);
end

%*************************************************
% Initialization of alpha, beta
%*************************************************
beta = ones(nView, 1)/nView;
alpha = ones(nView, 1)/nView;

wLHs = cell(1, nView);
for iView = 1:nView
    wLH = zeros(nSmp, size(Hs{iView}, 2));
    for iView2 = 1:nView
        wLH = wLH + alpha(iView2) * LHs{iView, iView2};
    end
    wLHs{iView} = wLH;
end

iter = 0;
objHistory = [];
converges = false;
maxIter = 50;
while ~converges
    %***********************************************
    % Update Y
    %***********************************************
    wLHW = zeros(nSmp, nCluster);
    for iView = 1:nView
        wLHW = wLHW + (1/beta(iView)) * wLHs{iView} * Ws{iView}; % n d c m
    end
    [Y,ff,~,~,converge_f] = updateFF(wLHW,Y);
    Y2 = bsxfun(@times, Y, 1./(sqrt(sum(Y, 1))+eps));
    % obj2 = compute_obj(Hs, LHs, Y, Ws, alpha, beta);
%     [~, label] = max(wLHW, [], 2);
%     Y = full(sparse(1:nSmp, label, ones(nSmp, 1), nSmp, nCluster));
    % [U, ~, V]= svd(wLHW, 'econ');
    % Y = U * V';
    % Y = Y(:,1:nCluster);

    %***********************************************
    % Update Ws
    %***********************************************
    for iView = 1 : nView
        HLLH = wLHs{iView}' * wLHs{iView};
        HLLH = (HLLH + HLLH')/2;
        HLY = wLHs{iView}' * Y2;
        Ws{iView} = updateWnew(HLLH, HLY, Ws{iView});
    end
    % obj2 = compute_obj(Hs, LHs, Y, Ws, alpha, beta);
    
    %***********************************************
    % Update alpha
    %***********************************************
    A = zeros(nView, nView);
    b = zeros(nView, 1);
    for iView = 1 : nView
        Ai = zeros(nView, nView);
        bi = zeros(nView, 1);
        LHWs = cell(1, nView);
        for iView2 = 1 : nView
            LHWs{iView2} = LHs{iView, iView2} * Ws{iView}; % n d c m^2
        end
        for iView2 = 1 : nView
            for iView3 = iView2 : nView
                Ai(iView2, iView3) = sum(sum( LHWs{iView2}.* LHWs{iView3})); % n c m^3
            end
            bi(iView2) = sum(sum( LHWs{iView2} .* Y2));
        end
        Ai = max(Ai, Ai');
        A = A + (1/beta(iView)) * Ai;
        b = b + (1/beta(iView)) * bi;
    end
    opt = [];
    opt.Display = 'off';
    alpha_old = alpha;
    [alpha,fval,~,~] = quadprog(A, -b, [], [], ones(1, nView), 1, zeros(nView, 1), ones(nView, 1), alpha_old, opt);
    % obj2 = compute_obj(Hs, LHs, Y, Ws, alpha, beta);
    
    wLHs = cell(1, nView);
    for iView = 1:nView
        wLH = zeros(nSmp, size(Hs{iView}, 2));
        for iView2 = 1:nView
            wLH = wLH + alpha(iView2) * LHs{iView, iView2};
        end
        wLHs{iView} = wLH;
    end
    
    %***********************************************
    % Update beta
    %***********************************************
    es = zeros(nView, 1);
    for iView = 1 : nView
        wLHW = wLHs{iView} * Ws{iView}; % n d c m
        E = wLHW - Y2;
        es(iView) = sum(sum( E.^2 ));
    end
    beta = sqrt(es)/sum(sqrt(es));
    % obj2 = compute_obj(Hs, LHs, Y, Ws, alpha, beta);
    
    obj = sum(es./beta);
    objHistory = [objHistory; obj]; %#ok
    
    if iter > 2 && (abs((objHistory(iter-1)-objHistory(iter))/objHistory(iter-1))<1e-4)
        converges = 1;
    end
    
    if iter > maxIter
        converges = 1;
    end
    iter = iter + 1;
end
[~, Iabel] = max(Y, [], 2);
% Y_normalized = NormalizeFea(Y);
% Iabel = litekmeans(Y_normalized, nCluster, 'MaxIter', 100, 'Replicates', 10);
end

function [W, objHistory] = updateWnew(A, B, W)
%     min tr(W' A W) - 2 tr(W' B)
%     st W'W = I
n = size(A, 1);

if nargout > 1
    obj = trace(W' * A * W) - 2 * trace(W' * B);
    objHistory = obj;
end

iter = 0;
converges = false;
maxIter = 5;
tol = 1e-3;
max_iter = 10;
largest_eigenvalue = power_iteration(A, tol, max_iter);
% largest_eigenvalue = eigs(sparse(A), 1, 'largestreal');
Atau = largest_eigenvalue*eye(n) - A;
while ~converges
    W_old = W;
    AWB = 2 * Atau * W_old + 2 * B;
    [U,~,V] = svd(AWB, 'econ');
    W = U * V';
    val = norm(W - W_old,'inf');
    if nargout > 1
        obj = trace(W' * A * W) - 2 * trace(W' * B);
        objHistory = [objHistory; obj]; %#ok
    end
    
    if iter > 2 && abs(val) < 1e-3
        converges = 1;
    end
    if iter > maxIter
        converges = 1;
    end
    iter = iter + 1;
end

end

function largest_eigenvalue = power_iteration(A, tol, max_iter)
% A: Symmetric matrix
% tol: Tolerance for convergence
% max_iter: Maximum number of iterations

% Ensure the matrix is symmetric
if ~isequal(A, A')
    error('The matrix is not symmetric.');
end

% Initial guess for the eigenvector
n = size(A, 1);
b_k = rand(n, 1);

for k = 1:max_iter
    % Calculate the matrix-by-vector product Ab
    b_k1 = A * b_k;
    
    % Re-normalize the vector
    b_k1_norm = norm(b_k1);
    b_k = b_k1 / b_k1_norm;
    
    % Check for convergence
    if k > 1 && norm(b_k - b_k_prev) < tol
        break;
    end
    
    b_k_prev = b_k;
end

% Rayleigh quotient for the eigenvalue
largest_eigenvalue = (b_k' * A * b_k) / (b_k' * b_k);
end


function [obj, grad] = solveWnew(W, A, B)
%     min tr(W' A W) - 2 tr(W' B)
%     st W'W = I
%
AW = A * W;
grad = 2 * (AW - B);
obj1 = sum(sum(W .* AW));
obj2 = sum(sum(W .* B));
obj = obj1 - 2 * obj2;
end

function [F,ff,obj_F,changed,converge_f] = updateFF(U,F) % O(nc)
% Preliminary
[n,c] = size(F);
obj_F = zeros(11,1);           

ff = sum(F);                        % O(nc)
uf = sum(U.*F);                     % O(nc)

up = zeros(1,c);
for cc = 1:c                        % O(c)
    up(cc) = uf(cc)./sqrt(ff(cc));  % O(1)
end
obj_F(1) = sum(up);                 % objf

changed = zeros(10,1);
incre_F = zeros(1,c);
converge_f = true;
% Update
for iterf = 1:10                    % O(nct) t<5
    converged = true;
    for i = 1:n
        ui = U(i,:);
        [~,id0] = find(F(i,:)==1);
        for k = 1:c                          % O(c)
            if k == id0
                incre_F(k) = uf(k)/sqrt(ff(k)+eps) - (uf(k) - ui(k))/sqrt(ff(k)-1+eps);
            else
                incre_F(k) = (uf(k)+ui(k))/sqrt(ff(k)+1+eps) - uf(k)/sqrt(ff(k)+eps);
            end
        end

        [~,id] = max(incre_F);
        if id~=id0                           % O(1)
            converged = false;               
            changed(iterf) = changed(iterf)+1;
            F(i,id0) = 0; F(i,id) = 1;
            ff(id0) = ff(id0) - 1;           % id0 from 1 to 0, number -1
            ff(id)  = ff(id) + 1;            % id from 0 to 1，number +1
            uf(id0) = uf(id0) - ui(id0);
            uf(id)  = uf(id) + ui(id);
        end
    end
    if converged
        break;
    end

    for cc = 1:c
        up(cc) = uf(cc)/sqrt(ff(cc)+eps);
    end
    obj_F(iterf+1) = sum(up);

    err_obj_f = obj_F(iterf+1)-obj_F(iterf);
    if err_obj_f < 0
        converge_f = false;
    end
end
end

function [obj] = compute_obj(Hs, LHs, Y, Ws, alpha, beta)
nView = length(Hs);
nSmp = size(Hs{1}, 1);

wLHs = cell(1, nView);
for iView = 1:nView
    wLH = zeros(nSmp, size(Hs{iView}, 2));
    for iView2 = 1:nView
        wLH = wLH + alpha(iView2) * LHs{iView, iView2};
    end
    wLHs{iView} = wLH;
end

Y2 = bsxfun(@times, Y, 1./(sqrt(sum(Y, 1))+eps));

previous = zeros(nView, 1);
for iView = 1:nView
    pre = wLHs{iView} * Ws{iView} - Y2;
    previous(iView) = sum(sum(pre .* pre));
end
obj = sum((1./beta) .* previous);
end