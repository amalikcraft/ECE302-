%Ahmad Malik    4/5/22
%ECE302-1
%Keene
%Project 3

clc; clear; close all;


%% Question 1

%number of samples
N = 5e5;  
%number of observations 
obs = 5:15;
%alpha and lambda parameters
alpha = [0.35, 0.7, 1];
lambda = [0.35, 0.7, 1];

%generating the MSE, Bias, and Variance of each distribution given the
%parameters above. This is done using the two generating functions.
exponential_lambda_1 = generate_exponential(N, obs,lambda(1));
exponential_lambda_2 = generate_exponential(N, obs,lambda(2));
exponential_lambda_3 = generate_exponential(N, obs,lambda(3));
rayleigh_aplha_1 = generate_rayleigh(N,obs,alpha(1));
rayleigh_aplha_2 = generate_rayleigh(N,obs,alpha(2));
rayleigh_aplha_3 = generate_rayleigh(N,obs,alpha(3));

%Subplot that compares each distributions MSE
figure;
subplot(1,2,1);
plot(obs, exponential_lambda_1(1,:), obs, exponential_lambda_2(1,:), obs,exponential_lambda_3(1,:));
title("MSE for Exponential");
xlabel("Number of Observations");
ylabel("MSE");
legend("\lambda = 0.35" , "\lambda = 0.7" ,"\lambda = 1.0");
xlim([5,15]);
subplot(1,2,2);
plot(obs, rayleigh_aplha_1(1,:), obs, rayleigh_aplha_2(1,:), obs, rayleigh_aplha_3(1,:));
title("MSE for Rayleigh");
xlabel("Number of Observations");
ylabel("MSE");
legend("\alpha = 0.35" ,"\alpha = 0.7", "\alpha = 1.0");

%Subplot that compares each distributions Bias
figure;
subplot(1,2,1);
plot(obs, exponential_lambda_1(2,:), obs, exponential_lambda_2(2,:), obs,exponential_lambda_3(2,:));
title("Bias for Exponential");
xlabel("Number of Observations");
ylabel("Bias");
legend("\lambda = 0.35", "\lambda = 0.7", "\lambda = 1.0");
xlim([5,15]);
subplot(1,2,2);
plot(obs, rayleigh_aplha_1(2,:), obs, rayleigh_aplha_2(2,:), obs, rayleigh_aplha_3(2,:));
title("Bias for Rayleigh");
xlabel("Number of Observations");
ylabel("Bias");
legend("\alpha = 0.35", "\alpha = 0.7", "\alpha = 1.0");

%Subplot that compares each distributions Variance
figure;
subplot(1,2,1);
plot(obs, exponential_lambda_1(3,:), obs, exponential_lambda_2(3,:), obs,exponential_lambda_3(3,:));
title("Variance for Exponential");
xlabel("Number of Observations");
ylabel("Variance");
legend("\lambda = 0.35", "\lambda = 0.7", "\lambda = 1.0");
xlim([5,15]);
subplot(1,2,2);
plot(obs, rayleigh_aplha_1(3,:), obs, rayleigh_aplha_2(3,:), obs, rayleigh_aplha_3(3,:));
title("Variance for Rayleigh");
xlabel("Number of Observations");
ylabel("Variance");
legend("\alpha = 0.35", "\alpha = 0.7", "\alpha = 1.0");


%% Question 2

%loading data
load data.mat; 
size = (size(data,2)); 

%Calculating each distributions parameters 
exponential_parameter = size./ sum(data, 2);                  
rayleigh_parameter = sqrt(.5 * mean(data.^2, 2));

%Variance of data
data_variance = var(data);
disp("Variance of data is : " + data_variance);
%Variance(Exponential) = 1 / parameter^2
Variance_Exponential = 1 / exponential_parameter^2;
disp("Variance of Exponential Distribution with parameter " + exponential_parameter +" is :  " + Variance_Exponential);

%Variance(Rayleigh) = (4 - pi)/2 * parameter^2
Variance_Rayleigh = (4 - pi)/2 * rayleigh_parameter^2;   
disp("Variance of Rayleigh Distribution given parameter  " +  rayleigh_parameter +" is :  " + Variance_Rayleigh);

fprintf(['\nSince the variance of the data (%f) is more closer to the variance of a\ntheoretical Rayleigh distribution (%f),',... 
'then that of the variance of a theoretical \nExponential disitribution (%f), then data was most likely,',...
' drawn from a Rayleigh distribution'], data_variance,Variance_Rayleigh, Variance_Exponential);

     
%% Functions

%This function generates a matrix that contains the MSE, bias, and variance
%of an exponential disitribution given the the number of samples,
%number of observations, and a lambda value.
function [Matrix] = generate_exponential(N,obs,lambda)
    len = length(obs);
    Matrix = zeros(3,len);
    for i = 1 : len
        distribution = exprnd(1/lambda, [N,(obs(i))]); %creating a exponential distribution 
        lambda_hat =  obs(i)./ sum(distribution ,2);  %lambda_hat
        Matrix(1,i) = mean((lambda - lambda_hat).^2); %MSE 
        Matrix(2,i) = mean(lambda_hat) - lambda; %Bias
        Matrix(3,i) = var(lambda_hat); %Variance
    end
end

%This function generates a matrix that contains the MSE, bias, and variance
%of a Rayleigh disitribution given the the number of samples,
%number of observations, and a lambda value.
function [Matrix] = generate_rayleigh(N,obs,alpha)
    len = length(obs);
    Matrix = zeros(3,len);
    for i = 1 : len
        rayleigh = raylrnd(alpha, [N,(obs(i))]); %creating a rayleigh distribution
        alpha_hat = sqrt(.5 *mean(rayleigh.^2,2));  %alpha_hat        
        Matrix(1,i) = mean((alpha - alpha_hat).^2); %MSE
        Matrix(2,i)  = mean(alpha_hat) - alpha; %Bias
        Matrix(3,i) = var(alpha_hat); %Variance
    end
end










