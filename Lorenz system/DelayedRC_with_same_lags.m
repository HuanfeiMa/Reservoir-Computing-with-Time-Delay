% Repreducing the dynamics prediction in Fig.2(a)
% with a delayed RC including 40 neurons with uniformly 5 lags for each
% neuron

clear all;clc
load('lorenzdata');


[Data, ps] = mapstd(data);% data normalization
resSize=40; % number of RC neurons


% load a typical set of random Win and Wres to reproduce the results in Fig.2
load UniformDelayedParameter; 
% or one can generate a random Win and Wres using following setting:
% arhow_r =0.67; % spectral radius
% d=0.05; % sparsity
% k=round(d*resSize);
% inSize=3;outSize=3;
% sigma=0.44;

dimension=5; % same lags for each neuron
tau=5;

gamma = 0.44; % leaky rate

initialen = 1000;
trainlen = 6000;
len = initialen+trainlen;
testlen = 3000;
r = zeros(resSize,len);

% training period
rtotal=zeros(resSize,len);
for i = 2:len
    ut = Data(:,i);
    r(:,i) = (1-gamma)*r(:,i-1) + gamma*(tanh( Win*ut + Wres*r(:,i-1)));
    rtotal(:,i) = r(:,i);
end
rtotal = rtotal(:,initialen:len-1);
traindata = Data(:,initialen+1:len);
rtrain=zeros(dimension*resSize,length(rtotal)-tau*dimension+1);

% neurons with lags
for k=1:resSize
    for i=1:dimension
        rtrain(i+dimension*(k-1),:)=rtotal(k,i*tau:end-dimension*tau+i*tau); 
    end
end
rrtrain=rtrain;
rrtrain(2:2:end,:)=rtrain(2:2:end,:).^2; % half neurons are nonlinear(even terms)

% Tikhonov regularization to solve Wout
traindata=traindata(:,tau*dimension:end);
beta = 1e-5; % regularization parameter
netsize=size(rrtrain,1);
Wout = ((rrtrain*rrtrain' + beta*eye(netsize)) \ (rrtrain*traindata(:,:)'))';
mse1=mean(mean((Wout*rrtrain-traindata).^2,2));

r2=zeros(1,resSize*dimension);
for k=1:resSize
    for i=1:dimension
        r2(i+dimension*(k-1))=r(k,end-dimension*tau+i*tau);
    end
end
r2(2:2:end) = r2(2:2:end).^2;

% testing period
vv =Wout*r2'; 
testoutput = zeros(3,testlen);
for i = len+1 : len+testlen
    ut = vv ; 
    testoutput(:,i)=vv;
    r(:,i) = (1-gamma)*r(:,i-1) + gamma*(tanh( Win*ut + Wres*r(:,i-1)));  
    for k=1:resSize
        for j=1:dimension
            r2(j+dimension*(k-1))=r(k,end-dimension*tau+j*tau);
        end
    end  
    r2(2:2:end) = r2(2:2:end).^2;
    vv = Wout * r2';   
end
testoutput(:,i)=vv;

original = Data(:,len+1:len+testlen);
predict = testoutput(:,len+1:len+testlen);

% plot
% 0.906--largest lyapunov exponent
t = (1:1:testlen)*0.01*0.906;
figure
subplot(3,1,1)
plot(t(1:1000),original(1,1:1000),'b','linewidth',1);
hold on
plot(t(1:1000),predict(1,1:1000),'r','linewidth',1);
xlim([0 9.06])
ylabel('x');
title('Lorenz system prediction');
subplot(3,1,2)
plot(t(1:1000),original(2,1:1000),'b','linewidth',1);
hold on
plot(t(1:1000),predict(2,1:1000),'r','linewidth',1);
ylabel('y');
xlim([0 9.06])
subplot(3,1,3)
plot(t(1:1000),original(3,1:1000),'b','linewidth',1);
hold on
plot(t(1:1000),predict(3,1:1000),'r','linewidth',1);
xlabel('$$\Lambda_{max}t$$', 'Interpreter', 'Latex')
ylabel('z');
legend('true data','Delayed RC')
xlim([0 9.06])