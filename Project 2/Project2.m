%Ahmad Malik    3/2/22
%ECE302-1
%Keene
%Project 2

clc; clear; close all;

%% Question 1

Y = 2*rand(1,1e6)-1; %Uniformly distributed random variable [-1,1]
W = 4*rand(1,1e6)-2; %Uniformaly distributed random variable [-2,2]
X = Y + W;  % Noisey Channel

%Conditional Mean y^(x)
y_hat = zeros(1,1e6);
%Generating y_hat using the provided piecewise function (8.30) for E[Y|X=x]
for i = 1:1e6         
    if X(i)>-1 && X(i)<1
        y_hat(i) = 0;
    end
    if X(i)< -1
        y_hat(i) = 0.5 + X(i)*.5;
    end
    if X(i)> 1
        y_hat(i) = -0.5 + X(i)*.5;
    end
end

%Computing the Bayesian MMSE
MMSE = mean((Y-y_hat).^2); % E[{Y-y^(x)}^2 |X]
bayesianMMSE = mean(MMSE);   % E[E[{Y-y^(x)}^2 |X]] // Averaged MMSE

%Computing the Linear MMSE
y_hat_linear = (1/5)*X; % Y^(x) Linear // From 8.6
linearMMSE = mean(mean((Y-y_hat_linear).^2)); % E[E[Y^(x)^2]]// Averaged LinearMMSE

%Generating Table
Method = {'Bayesian';'Linear'};
Theoretical = [1/4;4/15]; %Values calculated in the chapter 8.6
Experimental = [bayesianMMSE; linearMMSE]; 
Table = table(Theoretical,Experimental,'VariableNames',{'Theoretical','Experimental'},'RowNames',{'Bayesian','Linear'});
disp(Table);


%% Question 2

%Number of Observations
Num_Obs = 10;

%Computing The Theoretical MMSE given pairs of variances:

% Y and R have four variances
VarY = [0.25,0.5,0.75,1];
VarR = [1,0.75,0.5,.25];

%Generating Matrix for Theoretical MMSE
theoMMSE = zeros(Num_Obs,4); 
for i = 1: Num_Obs
    for j = 1:4
        theoMMSE(i,j) = (VarY(j) * VarR(j)) / (i * VarY(j) + VarR(j)); %Formula for MMSE given Var of Y and R
    end
end

%Computing Experimental Variance

%Using the mmseCalc function to calculate Experimental MMSE given the
%number of observations and the coressponding variance values of R and Y
for i = 1 : Num_Obs
    ExpMMSE1(i) = mmseCalc(i,VarY(1), VarR(1));
    ExpMMSE2(i) = mmseCalc(i,VarY(2), VarR(2));
    ExpMMSE3(i) = mmseCalc(i,VarY(3), VarR(3));
    ExpMMSE4(i) = mmseCalc(i,VarY(4), VarR(4));
end

% Generating subplots for each pair of Variances and comparing it the
% Theoretical and Experimental values

x = 1:1: Num_Obs;
subplot(2,2,1);
plot(x, ExpMMSE1(1, :), '-', x, theoMMSE(:, 1), 'X');
title("\sigma_{\it Y}^2 = " + VarY(1) + ", \sigma_{\it R}^2 = " + VarR(1));
xlabel("Number of Observations");
ylabel("MMSE");
legend("Theoretical","Experimental");
set(gcf, 'Position',  [100, 100, 1000, 800]);
ylim([-0.05 0.4]);

subplot(2,2,2);
plot(x, ExpMMSE2(1, :), '-', x, theoMMSE(:, 2), 'X');
title("\sigma_{\it Y}^2 = " + VarY(2) + ", \sigma_{\it R}^2 = " + VarR(2));
xlabel("Number of Observations");
ylabel("MMSE");
legend("Theoretical","Experimental");
set(gcf, 'Position',  [100, 100, 1000, 800])
ylim([-0.05 0.4]);

subplot(2,2,3);
plot(x, ExpMMSE3(1, :), '-', x, theoMMSE(:, 3), 'X');
title("\sigma_{\it Y}^2 = " + VarY(3) + ", \sigma_{\it R}^2 = " + VarR(3));
xlabel("Number of Observations");
ylabel("MMSE");
legend("Theoretical","Experimental");
set(gcf, 'Position',  [100, 100, 1000, 800])
ylim([-0.05 0.4]);

subplot(2,2,4);
plot(x, ExpMMSE4(1, :), '-', x, theoMMSE(:, 4), 'X');
title("\sigma_{\it Y}^2 = " + VarY(4) + ", \sigma_{\it R}^2 = " + VarR(4));
xlabel("Number of Observations");
ylabel("MMSE");
legend("Theoretical","Experimental");
set(gcf, 'Position',  [100, 100, 1000, 800])
ylim([-0.05 0.4]);

sgtitle('Theoretical & Experimental MMSE vs. Number of Observations')



%Function to find MMSE given the Variance of Y, Variance of R, and the Number of Observations

function [MMSE] = mmseCalc(obs,VarY, VarR)
    N = 1e6; %number of samples
    Y = normrnd(1, sqrt(VarY), [N 1]);   %Random Variable Y
    R = normrnd(0, sqrt(VarR), [N obs]); %Random Variable R and the number of observations
    X = zeros(N, obs);
    for i = 1:obs
        X(:, i) = R(:, i) + Y;   %X = Y + R
    end
    muY = mean(Y);  
    Exp_varY = var(Y);   % Variance of Y
    Exp_varR = zeros(N, obs);  
    for i = 1:obs
        Exp_varR(:, i) = X(:, i) - Y;    %Variance of R    
    end
    AvgVarR = var(reshape(Exp_varR, [], 1));     
    y_hat = (1 / (obs * Exp_varY + AvgVarR)) * (AvgVarR * muY + Exp_varY * sum(X, 2));  %fomula from reading
    MMSE = mean((Y - y_hat) .^ 2);  %compute MMSE using y^(x) and Y
end










