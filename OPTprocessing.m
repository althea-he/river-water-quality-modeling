% 遵循MATLAB的要求，所有的初始化和fmincon求解部分都被放置在脚本的顶部
% 而函数objectiveFunction被定义在了脚本的最后部分

% 第一阶段：获得最优化参数
tic % 开始计时
x = 1:30;
l = [24.04, 22.79, 22.10, 19.30, 18.98, 19.53, 18.62, 17.51, 16.39, 15.69, 14.33, 14.16, 13.68, 13.84, 12.85, 12.49, 11.43, 11.77, 10.44, 10.11, 9.82, 9.48, 9.51, 8.31, 8.25, 8.10, 7.39, 7.16, 6.98, 6.43];
d = [2.54, 3.25, 3.81, 4.06, 4.72, 4.92, 5.18, 5.53, 5.90, 6.08, 6.36, 6.27, 6.17, 6.78, 6.58, 6.51, 6.47, 7.02, 7.07, 6.55, 6.97, 6.98, 6.67, 6.31, 6.84, 6.67, 6.23, 6.23, 5.95, 5.89];
ni = [2.76, 2.67, 2.74, 2.57, 2.58, 2.48, 2.31, 2.23, 2.12, 2.08, 2.08, 1.90, 1.94, 1.86, 1.78, 1.77, 1.68, 1.59, 1.50, 1.51, 1.41, 1.42, 1.42, 1.29, 1.27, 1.20, 1.18, 1.17, 1.12, 1.05];
l0 = 24;
d0 = 2.02;
n0 = 3;
u = 36;
% 定义初始猜测和参数的上下界
initialGuess = [0.85, 2.25, 0.65, 0.75];
lb = [0.5, 1.5, 0.1, 0.2];
ub = [1.2, 3.0, 1.2, 1.3];
% 调用fmincon求解
options = optimoptions('fmincon', 'Display', 'iter', 'Algorithm', 'sqp');
[optParams, ztotalx] = fmincon(@(params) objectiveFunction(params, x, l, d, ni, l0, d0, n0, u), initialGuess, [], [], [], [], lb, ub, [], options);
% 打印最优化后的参数和ztotalx值
fprintf('最优化参数值: kd = %f, ka = %f, ks = %f, kn = %f\n', optParams(1), optParams(2), optParams(3), optParams(4));
fprintf('ztotalx: %f\n', ztotalx);


% 第二阶段：验证最优化参数
% 使用第一阶段得到的最优参数
kd = optParams(1);
ka = optParams(2);
ks = optParams(3);
kn = optParams(4);
% 初始化ztotaly
ztotaly = 0;
for n = 2:2:30
    lc = l0 * exp(-(kd + ks) * x(n) / u);
    dc = d0 * exp(-ka * x(n) / u) + kd * l0 / (ka - kd - ks) * (exp(-(kd + ks) * x(n) / u) - exp(-ka * x(n) / u)) + kn * n0 / (ka - kn) * (exp(-kn * x(n) / u) - exp(-ka * x(n) / u));
    nc = n0 * exp(-kn * x(n) / u);
    zl = (lc - l(n))^2;
    zd = (dc - d(n))^2;
    zn = (nc - ni(n))^2;
    ztotaly = ztotaly + zl + zd + zn;
end
% 计算比值
ratio = ztotaly / ztotalx;
% 打印结果
fprintf('ztotaly: %f\n', ztotaly);
fprintf('Ratio of ztotaly to ztotalx: %f\n', ratio);


% 第三阶段：目标断面浓度的计算
% 使用第一阶段得到的最优参数
lc = l0 * exp(-(kd + ks) * 45 / u);
dc = d0 * exp(-ka * 45 / u) + kd * l0 / (ka - kd - ks) * (exp(-(kd + ks) * 45 / u) - exp(-ka * 45 / u)) + kn * n0 / (ka - kn) * (exp(-kn * 45 / u) - exp(-ka * 45 / u));
nc = n0 * exp(-kn * 45 / u);
% 输出"x=45km时，lc的值、dc的值、nc的值"
fprintf('在x=45km时，BOD5浓度lc为：%f\n', lc);
fprintf('在x=45km时，氧亏值dc为：%f\n', dc);
fprintf('在x=45km时，有机氮浓度nc为：%f\n', nc);
% 根据浓度是否达标输出相应信息
if lc <= 4
    disp('BOD5浓度达标');
else
    disp('BOD5浓度不达标');
end

if dc <= 4.08
    disp('氧亏值达标');
else
    disp('氧亏值不达标');
end

if nc <= 0.50
    disp('有机氮浓度达标');
else
    disp('有机氮浓度不达标');
end


%第四阶段：作图
figure; 
% 使用第一阶段得到的最优参数
kd = optParams(1);
ka = optParams(2);
ks = optParams(3);
kn = optParams(4);
% 训练集
scatter(x(1:2:30), l(1:2:30), 'o', 'filled'); % BOD浓度，训练集
hold on;
scatter(x(1:2:30), d(1:2:30), 's', 'filled'); % 氧亏，训练集
scatter(x(1:2:30), ni(1:2:30), 'd', 'filled'); % 有机氮，训练集
% 验证集
scatter(x(2:2:30), l(2:2:30), 'x'); % BOD浓度，验证集
scatter(x(2:2:30), d(2:2:30), '+'); % 氧亏，验证集
scatter(x(2:2:30), ni(2:2:30), '*'); % 有机氮，验证集
% 可以计算并绘制模型预测的曲线
% 生成一个连续的x轴数据集，用于绘制平滑曲线，并确保范围至少到50km
x_continuous = linspace(0, 50, 100); % 这里将最大值设定为50km
l_pred = l0 * exp(-(kd + ks) * x_continuous / u);
d_pred = d0 * exp(-ka * x_continuous / u) + kd * l0 / (ka - kd - ks) * (exp(-(kd + ks) * x_continuous / u) - exp(-ka * x_continuous / u)) + kn * n0 / (ka - kn) * (exp(-kn * x_continuous / u) - exp(-ka * x_continuous / u));
ni_pred = n0 * exp(-kn * x_continuous / u);
% 绘制模型预测曲线
plot(x_continuous, l_pred, 'r-');
plot(x_continuous, d_pred, 'b-');
plot(x_continuous, ni_pred, 'g-');
% 设置图形属性
xlabel('断面距离（km）');
ylabel('浓度（mg/L）');
xlim([0, 50]); % 确保 x 轴至少大于 50 km
legend('BOD - 训练集', '氧亏 - 训练集', '有机氮 - 训练集', 'BOD - 验证集', '氧亏 - 验证集', '有机氮 - 验证集', 'Location', 'best');
title('污染物浓度随断面距离的变化');
hold off;

% 结束计时
toc;

% 在脚本最后定义目标函数objectiveFunction
function ztotalx = objectiveFunction(params, x, l, d, ni, l0, d0, n0, u)
    kd = params(1);
    ka = params(2);
    ks = params(3);
    kn = params(4);
    
    ztotalx = 0;
    for n = 1:2:30
        lc = l0 * exp(-(kd + ks) * x(n) / u);
        dc = d0 * exp(-ka * x(n) / u) + kd * l0 / (ka - kd - ks) * (exp(-(kd + ks) * x(n) / u) - exp(-ka * x(n) / u)) + kn * n0 / (ka - kn) * (exp(-kn * x(n) / u) - exp(-ka * x(n) / u));
        nc = n0 * exp(-kn * x(n) / u);
        zl = (lc - l(n))^2;
        zd = (dc - d(n))^2;
        zn = (nc - ni(n))^2;
        ztotalx = ztotalx + zl + zd + zn;
    end
end