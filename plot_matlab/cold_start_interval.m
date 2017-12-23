% ==========================================================================
clear;
clc;

% ��������
load data               % 
disp('Loading: data.mat');
lines = {
    'd-', ...
    'x-',  ...
    '<-', 's-', '*-', '^-'};
colors = {
    [0 0 0.5], ...
    [1 0.2 0],  ...
    [0 0 1],    [0 1 1],    [1 0 0],    [0 0.5 1]};
sub = [
    121, 122];
multiple = 100;
a = 't5';       % 'taobao'ƴд��ȷ��Ϊ����t3���ݿ⣬��������a1���ݿ�

if strcmp(a, 't5')
    dataset = t5_test_interval;
    ylims = {       % y�ᶥ��������������Ȼlegend��ѹס�̶�ֵ
    [-130, 250]
    [-10, 45]};
else
    dataset = a5_test_interval;
    ylims = {       % y�ᶥ��������������Ȼlegend��ѹס�̶�ֵ
    [0.5, 50000]
    [-10, 65]};    
end

figure();
set(gca,'FontSize',15);
x = data_interval_idxs;
set(gca, 'XTick', x);   % ָ��x��̶ȱ�ʶ
    
for num = [1, 2]
    name = data_evaluation_growth_rate{num};    % ����ָ��
    data = dataset{num};

    subplot(sub(num));
    for i = [1, 2, 3, 4, 5, 6]
        % ��������log��������amazon��recall@30������������ʱ
        if ~strcmp(a, 't5') && strcmp(name, data_evaluation_growth_rate{1})   
            %  a5 - Recall@30 (%)����log����
            semilogy(x, data{i} * multiple, ...
                lines{i}, 'Color',colors{i}, 'LineWidth', 2, 'MarkerSize', 8);
        else
            % ���������������(%)����
            plot(x,data{i} * multiple, ...
                lines{i}, 'Color',colors{i}, 'LineWidth', 2, 'MarkerSize', 8);            
        end        
        hold on;    % �Ȼ��ߣ���hold on
    end
    
    xlabel('interval');
    labels = data_interval_labels;    
    set(gca, 'XTick', x, 'XTickLabel', labels);   % ָ��x����ʾ��ʶ  
    xlim([0.5 10.5])
    ylabel(name)
    ylim(ylims{num})
end

%hl = legend(data_method_growth_rate);       % ���ַ�����
%set(hl, 'Location', 'NorthOutside', 'Orientation', 'horizontal','Box', 'on');





