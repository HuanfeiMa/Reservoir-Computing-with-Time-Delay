% Repreducing the dynamics prediction of the chaotic gene regulation model
% with a single neuron RC with 600 lags
clear all;clc

% generate data
lags=[18,18.25];
history=20;
tspan=0.1:0.1:5000;
sol=dde23(@ddefun,lags,history,tspan);
data = deval(sol,tspan);
data=data';

[Data, ps] = mapstd(data');% data normalization


% reservoir parameter
dimension=600;
resSize=1;
gamma = 0.1;  % leaky rate
tau=5;


% weight 
Win = 0.8;
Wres = 0.6;


initialen = 10000;
trainlen = 10000;
len = initialen+trainlen;
testlen = 10000;
r = zeros(resSize,len);

%training period
rtotal=zeros(resSize,len);
for i = 2:len
    ut = Data(i);
    r(i) = (1-gamma)*r(i-1) + gamma*(tanh( Win*ut + Wres*r(i-1)) );
    rtotal(i) = r(i);
end

rtotal = rtotal(initialen:len-1);
traindata = Data(initialen+1:len);
rtrain=zeros(dimension,length(rtotal)-tau*dimension+1);

%neurons with lags
for i=1:dimension
    rtrain(i,:)=rtotal(i*tau:end-dimension*tau+i*tau); 
end

rrtrain=rtrain;
rrtrain(2:2:end,:)=rtrain(2:2:end,:).^2; % half neurons are nonlinear(even terms)


% Tikhonov regularization to solve Wout   
traindata=traindata(tau*dimension:end);
beta = 1e-8; % regularization parameter
netsize=size(rrtrain,1);
Wout = ((rrtrain*rrtrain' + beta*eye(netsize)) \ (rrtrain*traindata(:,:)'))';
mse1=mean((Wout*rrtrain-traindata).^2,2);



for i=1:dimension
    r2(i)=r(end-dimension*tau+i*tau);
end
r2(2:2:end) = r2(2:2:end).^2;

trainoutput=Wout*rrtrain;
vv =Wout*r2'; 
testoutput = zeros(3,testlen);

% testing period
for i = len+1 : len+testlen
    ut = vv ;
    testoutput(:,i)=vv;
    r(i) = (1-gamma)*r(i-1) + gamma*(tanh( Win*ut + Wres*r(i-1) ));
    for j=1:dimension
    r2(j)=r(end-dimension*tau+j*tau);
    end
    r2(2:2:end) = r2(2:2:end).^2;
    vv = Wout * r2';
end
testoutput(:,i)=vv;

original = Data(:,len+1:len+testlen);
predict = testoutput(:,len+1:len+testlen);


% plot
t = (1:1:testlen)*0.1*0.033;
figure
plot(t(1:3000),original(1,1:3000),'b','linewidth',1);
hold on
plot(t(1:3000),predict(1,1:3000),'r','linewidth',1);
title('prediction')
xlabel('$$\Lambda_{max}t$$', 'Interpreter', 'Latex')
ylabel('x');
legend('true data','DelayedRC1*600')
