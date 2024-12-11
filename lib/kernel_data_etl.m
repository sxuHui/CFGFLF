function [Ks, Y] = kernel_data_etl(filename, filetype)
if ~exist('filetype', 'var')
    clear X Ks y Y;
    load(filename);
    if exist('X', 'var')
        filetype = 1;
    elseif exist('KH', 'var')
        filetype = 2;
    end
    clear X Ks y Y;
end
switch filetype
    case 1
        clear X y Y;
        load(filename);
        if exist('y', 'var')
            Y = y;
        end
        if size(X, 1) ~= size(Y, 1)
            Y = Y';
        end
        assert(size(X, 1) == size(Y, 1));
        Xs = cell(1,1);
        Xs{1} = X;
        Ks = Xs_to_Ks_12k(Xs);
        Ks2 = Ks{1,1};
        Ks = Ks2;
        clear Ks2 Xs;
        [nSmp, ~, nKernel] = size(Ks);
        Ks = kcenter(Ks);
        Ks = knorm(Ks);
    case 2
        clear KH Y;
        load(filename, 'KH', 'Y');
        Ks = KH; clear KH;
        [~, ~, Y] = unique(Y);
        Ks = kcenter(Ks);
        Ks = knorm(Ks);
    case 3
        %     RG-MVC-Robust Graph-based Multi-view Clustering-AAAI-2022
        clear KH Y;
        load(filename, 'KH', 'Y');
        Ks = KH; clear KH;
        Ks = remove_large(Ks);
        Ks = knorm(Ks);
        Ks = kcenter(Ks);
        Ks = divide_std(Ks);
end
