% Repreducing the dynamics prediction in SI
% with a standard non-delayed RC of dimension 600
clear all;clc
load('MGdata');


[Data, ps] = mapstd(data');% data normalization
resSize=600;% number of RC neurons

% load a typical set of random Win and Wres to reproduce the results
load NondelayedParameter; 
% or one can generate a random Win and Wres using following setting:
% arhow_r =0.79; % spectral radius
% d = 0.2; % sparsity
% k=round(d*resSize);
% inSize = 1; outSize = 1;
% sigma = 0.44;

gamma = 0.44; % leaky rate


initialen = 10000;
trainlen = 25000;
len = initialen+trainlen;
testlen = 6000;
r = zeros(resSize,1);

rtotal=zeros(resSize,len);

% training period
for i = 1:len
    ut = Data(i);
    r = (1-gamma)*r + gamma*(tanh( Win*ut + Wres*r));
    rtotal(:,i) = r;
end
rtotal = rtotal(:,initialen:len-1);
rtrain = rtotal; 
rtrain(2:2:end,:)=rtrain(2:2:end,:).^2; % half neurons are nonlinear(even terms)

% Tikhonov regularization to solve Wout
traindata = Data(initialen+1:len);    
beta = 1e-4; % regularization parameter
Wout = ((rtrain*rtrain' + beta*eye(resSize)) \ (rtrain*traindata'))';
trainoutput = Wout*rtrain;    
mse1 = mean((trainoutput-traindata).^2,2);


% testing period
vv = trainoutput(:,trainlen);
r=rtotal(:,end);
testoutput = [ ];
for i = 1 : testlen
    ut = vv ; 
    r = (1-gamma)*r + gamma*(tanh( Win*ut + Wres*r));
    r2 = r;
    r2(2:2:end) = r2(2:2:end).^2;
    vv = Wout * r2;
    testoutput = [testoutput vv];
end
testdata = Data(len+1:len+testlen);

% plot
% 0.006--largest lyapunov exponent
t = (1:1:testlen)*0.5*0.006;
figure
plot(t(1:1000),testdata(1,1:1000),'b','linewidth',1);
hold on
plot(t(1:1000),testoutput(1,1:1000),'r','linewidth',1);
title('MG system prediction')
xlabel('$$\Lambda_{max}t$$', 'Interpreter', 'Latex')
ylabel('x');
legend('true data','RC')