%Ahmad Malik
%ECE302-1
%Project 5

clc; clear; close all;


%{
In this project, an iid signal s[n] is sent through a filter c[n] which is 
then combined with zero mean gaussian noise d[n]. Thus we recieve a signal 
r[n]. To recover signal s[n], we pass r[n] through a wiener filter h[n] to 
get s^[n] which is an estimate of the original signal s[n]. To find the 
best wiener filter given it's length, we must compute the normal equations 
that require finding the cross and auto correlations of the sent and
recieved signal. We can measure how well the filter performs by computing 
the MSE.
%}


%% Part 2

C = [1, 0.2, 0.4]; % C[n]
N = [4, 6, 10]; %Length of filter
variance = .5; %variance
sigma = sqrt(variance); %covariance
mu = 0; 

%discrete random signal +/- 1
s = randi(2, [1,1e6]);
s = -1*double(s==2) + double(s==1);

%output of first filter:  r = filter{s} + d
r = filter(C, 1, s) + normrnd(mu, sigma , 1, 1e6);

%MSE for N = 4,6,10
MSE = zeros(1,3);
MSE(1) = wienerMSE(s,r,N(1));
MSE(2) = wienerMSE(s,r,N(2));
MSE(3) = wienerMSE(s,r,N(3));

T = table([MSE(1); MSE(2); MSE(3)], 'RowNames', {'N=4', 'N=6', 'N=10'});
T.Properties.VariableNames = ("MSE")

%MSE function
function MSE = wienerMSE(s,r,N)
    Rsr = zeros(N, 1);
    Rrr = zeros(N, 1);
    for i = 1:N
        %cross correlation
        Rsr(i) = mean(s(i:end) .* r(1:end + 1 - i));
        %auto correlation
        Rrr(i) = mean(r(i:end) .* r(1:end + 1 - i));
    end
    %generating matrix for normal equations
    Rrr_Matrix = toeplitz(Rrr);

    %solve for h 
    h = inv(Rrr_Matrix)* Rsr; 

    %computing s^[n]
    s_hat = filter(h , 1, r);

    %computing MSE
    MSE = mean((s - s_hat) .^ 2);
end

