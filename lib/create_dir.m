function create_dir(dir_name)
try
    if ~exist(dir_name, 'dir'); mkdir(dir_name); end
catch
    disp(['create dir: ',dir_name, 'failed, check the authorization']);
end
end