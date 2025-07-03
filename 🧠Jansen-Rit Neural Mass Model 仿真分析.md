# 🧠Jansen-Rit Neural Mass Model 仿真分析

##  模型简介

Jansen-Rit 模型（Jansen & Rit, 1995）模拟的是大脑皮层一个功能柱的神经群体动力学。模型由三种神经元群体组成：

- 主锥体神经元（Pyramidal neurons）
- 兴奋性中间神经元（Excitatory interneurons）
- 抑制性中间神经元（Inhibitory interneurons）

---

##  Step 1：参数定义

这些参数定义了神经元之间的连接强度、突触增益和动力学特性：

```matlab
A = 3.25;     % 兴奋性传递增益（mV）
B = 22;       % 抑制性传递增益（mV）
C = 135;      % 总连接强度

v0 = 6;       % Sigmoid阈值电压（mV）
e0 = 2.5;     % Sigmoid最大值（Hz）
R  = 0.56;    % Sigmoid斜率

a = 100;      % 兴奋性逆时常数（s^-1）
b = 50;       % 抑制性逆时常数（s^-1）

C1 = C; C2 = 0.8*C; C3 = C/4; C4 = C/4;
```

## Step 2：输入信号（模拟外部刺激）

```matlab
fs = 2000;                     % 采样率 Hz
dt = 1/fs;                     % 时间间隔
t = 0:dt:6;                    % 仿真时间（6秒）

MEAN = 220; SD = 22;
P_in = normrnd(MEAN, SD, size(t));  % 高斯噪声输入
```

## Step 3：状态变量与模型结构

模型包含 6 个状态变量，对应 3 类神经元群体的突触后电位 y(t)y(t)y(t) 与其一阶导数 $$\dot{y}(t)$$：

- $$y_0, \dot{y}$$：主锥体输出
- $$y_1, \dot{y}_1$$：兴奋性反馈
- $$y_2, \dot{y}_2$$：抑制性反馈

## Step 4：神经动力学微分方程

每个突触后电位满足如下二阶微分方程：

$$\frac{d^2y(t)}{dt^2} = A \cdot a \cdot S(v) - 2a \cdot \frac{dy(t)}{dt} - a^2 \cdot y(t)$$

- A：增益（或 B）
- a：时间常数（或 b）
- S(v)：Sigmoid 激活函数（脉冲发放率）

## Step 5：Runge-Kutta 四阶积分法（RK4）


$$
\begin{aligned}
k_1 &= f(y_t, t) \\\\
k_2 &= f\left(y_t + \frac{h}{2}k_1,\, t + \frac{h}{2} \right) \\\\
k_3 &= f\left(y_t + \frac{h}{2}k_2,\, t + \frac{h}{2} \right) \\\\
k_4 &= f\left(y_t + h\,k_3,\, t + h \right) \\\\
y_{t+h} &= y_t + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)
\end{aligned}
$$






































