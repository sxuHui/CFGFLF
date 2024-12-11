function avg_std_latex(latex_file_name, avg_table, std_table, row_labels, column_labels, format)
% 检查format参数，确定小数位数
num_decimal_places = str2double(format);
if isnan(num_decimal_places) || num_decimal_places < 0 || num_decimal_places > 10
    error('format参数必须是一个介于0到10之间的数值。');
end

% 定义小数位数格式化字符串
decimal_format = ['%.' num2str(num_decimal_places) 'f'];

% 打开文件用于写入
fid = fopen(latex_file_name, 'w');
if fid == -1
    error('无法创建或打开文件 %s。', latex_file_name);
end

% 打印LaTeX表格的开始部分
% fprintf(fid, '\\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|}\n');
% fprintf(fid, '\\hline\n');

num_columns = size(avg_table, 2) + 1;

% 构建表头字符串
header_string = '';
for i = 1:num_columns
    header_string = [header_string, 'c']; % 每列都是居中对齐
    if i < num_columns
        header_string = [header_string, ' ']; % 列与列之间以空格分隔
    end
end

% 打印 LaTeX 表格的开始部分和表头
fprintf(fid, '\\begin{tabular}{%s}\n', header_string);
fprintf(fid, '\\toprule\n');

% 打印列标签
fprintf(fid, ' & ');
for i = 1:length(column_labels)
    if i == length(column_labels)
        fprintf(fid, '%s ', column_labels{i});
    else
        fprintf(fid, '%s & ', column_labels{i});
    end
end
fprintf(fid, ' \\\\ \n\\midrule\n');

% 打印行标签和数据
for i = 1:size(avg_table, 1)
    fprintf(fid, '%s & ', row_labels{i});
    for j = 1:size(avg_table, 2)
        % 格式化平均值和标准差，保留指定的小数位数
        avg_val = sprintf(decimal_format, avg_table(i, j));
        std_val = sprintf(decimal_format, std_table(i, j));
        % 打印每个单元格的内容，使用format参数确定小数位数
        if j == size(avg_table, 2)
            fprintf(fid, '%s ${\\pm}$ %s', avg_val, std_val);
        else
            fprintf(fid, '%s ${\\pm}$ %s & ', avg_val, std_val);
        end
    end
    if i == size(avg_table, 1)
        fprintf(fid, ' \\\\ \n\\bottomrule\n');
    else
        fprintf(fid, ' \\\\ \n\\midrule\n');
    end    
end

% 打印LaTeX表格的结束部分
fprintf(fid, '\\end{tabular}');
fclose(fid);
end