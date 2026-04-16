% 第一阶段：初始化
tic % 开始计时
% 初始化参数和数组
x = 1:30;
l = [24.04, 22.79, 22.10, 19.30, 18.98, 19.53, 18.62, 17.51, 16.39, 15.69, 14.33, 14.16, 13.68, 13.84, 12.85, 12.49, 11.43, 11.77, 10.44, 10.11, 9.82, 9.48, 9.51, 8.31, 8.25, 8.10, 7.39, 7.16, 6.98, 6.43];
d = [2.54, 3.25, 3.81, 4.06, 4.72, 4.92, 5.18, 5.53, 5.90, 6.08, 6.36, 6.27, 6.17, 6.78, 6.58, 6.51, 6.47, 7.02, 7.07, 6.55, 6.97, 6.98, 6.67, 6.31, 6.84, 6.67, 6.23, 6.23, 5.95, 5.89];
ni = [2.76, 2.67, 2.74, 2.57, 2.58, 2.48, 2.31, 2.23, 2.12, 2.08, 2.08, 1.90, 1.94, 1.86, 1.78, 1.77, 1.68, 1.59, 1.50, 1.51, 1.41, 1.42, 1.42, 1.29, 1.27, 1.20, 1.18, 1.17, 1.12, 1.05];
l0 = 24;
d0 = 2.02;
n0 = 3;
u = 36;
accept = zeros(71,151,111,111);
find = zeros(132093441,4); % 132093441是最大值，后续根据需要调整
count = 0; % count用于跟踪满足条件的参数组合数量


% 第二阶段：主循环
% 通过四层嵌套循环遍历一系列参数组合（kd、ka、ks、kn），计算特定条件下的模型预测值，并与实际测量值进行比较
for i = 1:71
    kd = 0.49 + 0.01*i;
    for j = 1:151
        ka = 1.49 + 0.01*j;
        for k = 1:111
            ks = 0.09 + 0.01*k;
            for m = 1:111
                kn = 0.19 + 0.01*m;
                accept(i,j,k,m)=1; % 初始假设为接受状态
                for n = 1:2:30 % 取奇数断面距离为训练集
                    % 如果ka=kd+ks或ka=kn，则微分方程无解，当前参数组合被排除
                    if (ka == kd + ks || ka == kn)
                        accept(i,j,k,m)=0;
                        break;
                    end
                    lc = l0 * exp(-(kd + ks) * x(n) / u);
                    dc = d0 * exp(-ka * x(n) / u) + kd * l0 / (ka - kd - ks) * (exp(-(kd + ks) * x(n) / u) - exp(-ka * x(n) / u)) + kn * n0 / (ka - kn) * (exp(-kn * x(n) / u) - exp(-ka * x(n) / u));
                    nc = n0 * exp(-kn * x(n) / u);
                    % 如果任一计算值与实际值的差异超过10%，则当前参数组合同样被排除
                    if (abs(lc - l(n)) > 0.1 * l(n) || abs(dc - d(n)) > 0.1 * d(n) || abs(nc - ni(n)) > 0.1 * ni(n))
                        accept(i,j,k,m)=0;
                        break;
                    end
                end
                if accept(i,j,k,m)~=0
                    count = count + 1;
                    find(count,:) = [i, j, k, m];
                end
            end
        end
    end
end

% 调整find数组的大小以匹配实际计数值
find = find(1:count, :);
% 输出和保存结果
disp(['被接受的参数组共', num2str(count), '组']);
save('accept.mat', 'accept');
save('find.mat', 'find');


% 检查find数组中的参数是否真正符合条件
num_e = count * 15;
le = zeros(num_e, 1);
de = zeros(num_e, 1);
ne = zeros(num_e, 1);

error_idx = 0;  % 误差数组的索引

for idx = 1:count
    kd = 0.49 + 0.01 * (find(idx, 1) - 1);
    ka = 1.49 + 0.01 * (find(idx, 2) - 1);
    ks = 0.09 + 0.01 * (find(idx, 3) - 1);
    kn = 0.19 + 0.01 * (find(idx, 4) - 1);
   
    for n = 1:2:30 % 奇数断面距离为训练集
        x_n = x(n); 
        exp_kd_ks_x_u = exp(-(kd + ks) * x_n / u);
        exp_ka_x_u = exp(-ka * x_n / u);
        exp_kn_x_u = exp(-kn * x_n / u);
        
        lc = l0 * exp_kd_ks_x_u;
        dc = d0 * exp_ka_x_u + kd * l0 / (ka - kd - ks) * (exp_kd_ks_x_u - exp_ka_x_u) + kn * n0 / (ka - kn) * (exp_kn_x_u - exp_ka_x_u);
        nc = n0 * exp_kn_x_u;

        error_idx = error_idx + 1;
        le(error_idx) = abs(lc - l(n)) / l(n);
        de(error_idx) = abs(dc - d(n)) / d(n);
        ne(error_idx) = abs(nc - ni(n)) / ni(n);
    end
end

% 保存误差值到check.mat文件
save('check.mat', 'le', 'de', 'ne');
% check.mat文件夹中的所有误差值均小于0.1，证明第二阶段没有问题


% 第三阶段：作图——kd、ka、ks、kn的后验分布图
% kd的后验分布图
figure
n = 1:71;
x = 0.49 + 0.01*n; 
y = arrayfun(@(n) sum(sum(sum(accept(n,:,:,:)))), n);
y = y/count; 
plot(x, y, 'b-', 'LineWidth', 1); 
xlim([0.5, 1.2]);
xlabel("kd(1/d)"); 
ylabel("PDF"); 
title('kd的后验分布图'); 
%ka的后验分布图
figure
n = 1:151;
x = 1.49 + 0.01*n;
y = arrayfun(@(n) sum(sum(sum(accept(:,n,:,:)))), n);
y = y/count; 
plot(x, y, 'b-', 'LineWidth', 1); 
xlim([1.5, 3.0]);
xlabel("ka(1/d)"); 
ylabel("PDF"); 
title('ka的后验分布图'); 
%ks的后验分布图
figure
n = 1:111;
x = 0.09 + 0.01*n;
y = arrayfun(@(n) sum(sum(sum(accept(:,:,n,:)))), n);
y=y/count;
plot(x, y, 'b-', 'LineWidth', 1); 
xlim([0.1, 1.2]);
xlabel("ks(1/d)"); 
ylabel("PDF"); 
title('ks的后验分布图'); 
%kn的后验分布图
figure
n = 1:111;
x = 0.19 + 0.01*n;
y = arrayfun(@(n) sum(sum(sum(accept(:,:,:,n)))), n);
y=y/count;
plot(x, y, 'b-', 'LineWidth', 1); 
xlim([0.2, 1.3]);
xlabel("kn(1/d)"); 
ylabel("PDF"); 
title('kn的后验分布图'); 


% 第四阶段：对通过接受条件的参数组用验证集进行误差计算
% 简化过程：假设 le_errors, de_errors, ne_errors 的大小已知，这里使用 count * 15，因为 n 从 2 到 30 步长为 2
x = 1:30;  %第三阶段作图时改变了最初x的定义，在第四阶段需要改回原来的x数组！！！特别注意变量的使用！！！
num_errors = count * 15;
le_errors = zeros(num_errors, 1);
de_errors = zeros(num_errors, 1);
ne_errors = zeros(num_errors, 1);

error_idx = 0;  % 误差数组的索引

for idx = 1:count
    kd = 0.49 + 0.01 * (find(idx, 1) - 1);
    ka = 1.49 + 0.01 * (find(idx, 2) - 1);
    ks = 0.09 + 0.01 * (find(idx, 3) - 1);
    kn = 0.19 + 0.01 * (find(idx, 4) - 1);

    for n = 2:2:30
        x_n = x(n); 
        exp_kd_ks_x_u = exp(-(kd + ks) * x_n / u);
        exp_ka_x_u = exp(-ka * x_n / u);
        exp_kn_x_u = exp(-kn * x_n / u);
        
        lc = l0 * exp_kd_ks_x_u;
        dc = d0 * exp_ka_x_u + kd * l0 / (ka - kd - ks) * (exp_kd_ks_x_u - exp_ka_x_u) + kn * n0 / (ka - kn) * (exp_kn_x_u - exp_ka_x_u);
        nc = n0 * exp_kn_x_u;

        error_idx = error_idx + 1;
        le_errors(error_idx) = abs(lc - l(n)) / l(n);
        de_errors(error_idx) = abs(dc - d(n)) / d(n);
        ne_errors(error_idx) = abs(nc - ni(n)) / ni(n);
    end
end

% 保存由验证集得到的误差值到error.mat文件，便于核查

% 绘制误差的概率密度分布图
figure;
histogram(le_errors, 'Normalization', 'pdf');
title('BOD5相对误差le的分布');
xlabel('误差');
ylabel('PDF');

figure;
histogram(de_errors, 'Normalization', 'pdf');
title('氧亏相对误差de的分布');
xlabel('误差');
ylabel('PDF');

figure;
histogram(ne_errors, 'Normalization', 'pdf');
title('有机氮相对误差ne的分布');
xlabel('误差');
ylabel('PDF');


%第五阶段：统计与概率计算
%分析距离河段起始断面45km处的l、d、ni值是否低于特定阈值
% 初始化统计计数器
countL = 0; % 记录l值低于阈值的次数
countD = 0; % 记录d值低于阈值的次数
countN = 0; % 记录n值低于阈值的次数
countAll = 0; % 记录同时满足所有三个条件的次数

for zzz = 1:count
    i = find(zzz,1);
    j = find(zzz,2);
    k = find(zzz,3);
    m = find(zzz,4);
    kd = 0.49 + 0.01 * i;
    ka = 1.49 + 0.01 * j;
    ks = 0.09 + 0.01 * k;
    kn = 0.19 + 0.01 * m;
    lc = l0 * exp(-(kd + ks) * 45 / u);
    dc = d0 * exp(-ka * 45 / u) + kd * l0 / (ka - kd - ks) * (exp(-(kd + ks) * 45 / u) - exp(-ka * 45 / u)) + kn * n0 / (ka - kn) * (exp(-kn * 45 / u) - exp(-ka * 45 / u));
    nc = n0 * exp(-kn * 45 / u);
    
    % 统计各个条件的满足情况
    isLValid = lc <= 4;
    isDValid = dc <= 4.08;
    isNValid = nc <= 0.50;
    countL = countL + isLValid;
    countD = countD + isDValid;
    countN = countN + isNValid;
    countAll = countAll + (isLValid && isDValid && isNValid); %如果所有三个限值条件都满足，则countAll增加1
end

% 计算并输出概率
ProbL = countL / count;
ProbD = countD / count;
ProbN = countN / count;
Prob = countAll / count;
fprintf('l值低于阈值的概率：%f\n', ProbL);
fprintf('d值低于阈值的概率：%f\n', ProbD);
fprintf('n值低于阈值的概率：%f\n', ProbN);
fprintf('同时满足所有三个条件的概率：%f\n', Prob);

% 结束计时
toc;