function [LHs, LHLHs] = Hs2LHs_PPR(Hs, mu, m, knn_size)
nView = length(Hs);
nSmp = size(Hs{1}, 1);
B = cell(nSmp, m);

% Ss = cell(1, nView);
% Ls = cell(1, nView);
% HLs = cell(1, nView);
LHs = cell(nView, nView);
for iView = 1:nView
    [~, Xa] = litekmeans(Hs{iView}, m, 'Replicates', 1);   
    B = ConstructBP_pkn(Hs{iView}, Xa, 'nNeighbor', knn_size);
    idx = sum(B, 1) > 0;
    B = B(:, idx);
    P = B * diag(sum(B, 1).^(-.5));
    PTP = P' * P + 1e-5 * eye(size(P, 2));% m^3
    PTP = (PTP + PTP')/2;
    I_m = eye(size(B, 2));
    % core_PPR = inv((1+mu)/mu*I_m - PTP); % Inverse term (I - mu *  P' * P)^(-1)
    core_PPR_P = P /((1+mu)/mu*I_m - PTP);
    w0 = 1/(1+mu);
    if sum(sum(isnan(core_PPR_P))) + sum(sum(isinf(core_PPR_P))) > 0
        disp('NAN');
        idx_filter = [idx_filter; iView1]; %#ok
    end
    for iView2 = 1:nView
        tmp1 = P' * Hs{iView2}; % m * n * d
        LHs{iView2, iView} = w0 * Hs{iView2} + w0 * core_PPR_P * tmp1; % n m d
    end
end

clear S DS L HL;
if nargout > 1
    nDim = size(LHs{1,1}, 2);
    LHLHs = cell(nView, 1);
    for iView1 = 1:nView
        
        idx = 0;
        ABs = zeros((nDim*nDim), nView*(nView+1)/2);
        for iView2 = 1:nView
            LHa = LHs{iView1, iView2};
            for iView3 = iView2:nView
                LHb = LHs{iView1, iView3};
                idx = idx + 1;
                LHaLHb = LHa' * LHb;
                ABs(:, idx) = LHaLHb(:);
            end
        end
        LHLHs{iView1} = ABs;
    end
end

end