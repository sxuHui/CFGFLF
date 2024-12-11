function [K, t] = X2K_rmkkm_2015(X, kernel_id)
%
% The code of RMKKM [IJCAI 2015] is
%         PolynomialDegrees = [2, 4];
%         PolyPlusDegrees = [2, 4];
%         GaussianDegrees = [0.01, 0.05, 0.1, 1, 10, 50, 100];
% The paper of RMKKM [IJCAI 2015] is
%         PolynomialDegrees = [0, 1];
%         PolyPlusDegrees = [2, 4];
%         GaussianDegrees = [0.01, 0.05, 0.1, 1, 10, 50, 100];


switch kernel_id
    case 1
        %*********************************************
        % Linear Kernel
        %*********************************************
        t1_s = tic;
        K = full(X * X');
        % K = kcenter(K0);
        K = knorm_large(K);
        K = (K + K')/2;
        t = toc(t1_s);
    case 2
        %*********************************************
        % PolyPlus Kernel
        %*********************************************
        t1_s = tic;
        K = full(X * X');
        K = (K + 0).^2;
        % K = kcenter(K);
        K = knorm_large(K);
        K = (K + K')/2;
        t = toc(t1_s);
    case 3
        %*********************************************
        % PolyPlus Kernel
        %*********************************************
        t1_s = tic;
        K = full(X * X');
        K = (K + 0).^4;
        % K = kcenter(K);
        K = knorm_large(K);
        K = (K + K')/2;
        t = toc(t1_s);
    case 4
        %*********************************************
        % Polynomial Kernel
        %*********************************************
        t1_s = tic;
        K = full(X * X');
        K = (K + 1).^2;
        % K = kcenter(K);
        K = knorm_large(K);
        K = (K + K')/2;
        t = toc(t1_s);
    case 5
        %*********************************************
        % Polynomial Kernel
        %*********************************************
        t1_s = tic;
        K = full(X * X');
        K = (K + 1).^4;
        % K = kcenter(K);
        K = knorm_large(K);
        K = (K + K')/2;
        t = toc(t1_s);
    case 6
        %*********************************************
        % Gaussian Kernel
        %*********************************************
        t1_s = tic;
        K = EuDist2(X, [], 0);
        s = sqrt(max(max(K)));
        K = exp(-K / (2 * (0.01 * s)^2 ));
        % K = kcenter(K);
        K = knorm_large(K);
        K = (K + K')/2;
        t = toc(t1_s);
    case 7
        t1_s = tic;
        K = EuDist2(X, [], 0);
        s = sqrt(max(max(K)));
        K = exp(-K / (2 * (0.05 * s)^2 ));
        % K = kcenter(K);
        K = knorm_large(K);
        K = (K + K')/2;
        t = toc(t1_s);
    case 8
        t1_s = tic;
        K = EuDist2(X, [], 0);
        s = sqrt(max(max(K)));
        K = exp(-K / (2 * (0.1 * s)^2 ));
        % K = kcenter(K);
        K = knorm_large(K);
        K = (K + K')/2;
        t = toc(t1_s);
    case 9
        t1_s = tic;
        K = EuDist2(X, [], 0);
        s = sqrt(max(max(K)));
        K = exp(-K / (2 * (1 * s)^2 ));
        % K = kcenter(K);
        K = knorm_large(K);
        K = (K + K')/2;
        t = toc(t1_s);
    case 10
        t1_s = tic;
        K = EuDist2(X, [], 0);
        s = sqrt(max(max(K)));
        K = exp(-K / (2 * (10 * s)^2 ));
        % K = kcenter(K);
        K = knorm_large(K);
        K = (K + K')/2;
        t = toc(t1_s);
    case 11
        t1_s = tic;
        K = EuDist2(X, [], 0);
        s = sqrt(max(max(K)));
        K = exp(-K / (2 * (50 * s)^2 ));
        % K = kcenter(K);
        K = knorm_large(K);
        K = (K + K')/2;
        t = toc(t1_s);
    case 12
        t1_s = tic;
        K = EuDist2(X, [], 0);
        s = sqrt(max(max(K)));
        K = exp(-K / (2 * (0.01 * s)^2 ));
        % K = kcenter(K);
        K = knorm_large(K);
        K = (K + K')/2;
        t = toc(t1_s);
end