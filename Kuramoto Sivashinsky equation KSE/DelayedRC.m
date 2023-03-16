clear all;clc
load('KSEdata.mat');

% data normalization,means to 0 and deviations to 1
[data, ps] = mapstd(data);

tau1 = 0.25; % time step
M = 64;  % number of spatial grid points 

inSize = size(data,1);
resSize = 1000;
rho = 0.1; % spectral radius
gamma = 0.9; % leaky rate

dimension=5;
tau=1;

initialen = 10000;
trainlen = 20000;
% trainlen = 20000+tau*dimension;
len = initialen+trainlen;
testlen = 10000;


% generate weight matrix
Win1 =  -1 + 2*rand(resSize,inSize);
adj1 = zeros(resSize,inSize);
for m=1:resSize
    for n=1:inSize
        if(rand(1,1)<0.01)
            adj1(m,n)=1;
        end
    end
end
Win = adj1.*Win1;

sparsity = 0.006; % sparsity
k=round(sparsity*resSize);
adj2 = zeros(resSize,resSize);
for i = 1:resSize
    num = randperm(resSize,k);
    for j = 1:k
        adj2(i,num(j)) = 1;
    end
end
Wres1 = rand(resSize,resSize); 
Wres2 = adj2.*Wres1 ;
SR = max(abs(eig(Wres2))) ;
Wres = Wres2 .* ( rho/SR);

% training period
traindata=data(:,initialen+1:len);
states1 = zeros(resSize,len);
for i = 2:len
    ut = data(:,i);
    states1(:,i) = (1-gamma)*states1(:,i-1)+gamma*tanh(Wres*states1(:,i-1) + Win*ut);
end
states = states1(:,initialen:len-1);

rtrain=zeros(dimension*resSize,length(states)-tau*dimension+1);
%neurons with lags
for k=1:resSize
    for i=1:dimension
        rtrain(i+dimension*(k-1),:)=states(k,i*tau:end-dimension*tau+i*tau); 
    end
end
rrtrain=rtrain;
rrtrain(2:2:end,:)=rtrain(2:2:end,:).^2; % half neurons are nonlinear(even terms)

% Tikhonov regularization to solve Wout
traindata=traindata(:,tau*dimension:end);
netsize=size(rrtrain,1);
beta = 1e-4; % regularization parameter
Wout = ((rrtrain*rrtrain' + beta*eye(netsize)) \ (rrtrain*traindata(:,:)'))';
mse1=mean(sum((Wout*rrtrain-traindata).^2));

r2=zeros(1,resSize*dimension);
for k=1:resSize
    for i=1:dimension
        r2(i+dimension*(k-1))=states1(k,end-dimension*tau+i*tau);
    end
end
r2(2:2:end)=r2(2:2:end).^2;

%testing period
vv =Wout*r2'; 
testoutput = zeros(inSize,testlen);
for i = len+1 : len+testlen
    ut = vv ; 
    testoutput(:,i)=vv;
    states1(:,i) = (1-gamma)*states1(:,i-1)+gamma*tanh(Wres*states1(:,i-1) + Win*ut);
    for k=1:resSize
        for j=1:dimension
            r2(j+dimension*(k-1))=states1(k,end-dimension*tau+j*tau);
        end
    end
    r2(2:2:end)=r2(2:2:end).^2;  
    vv = Wout * r2';
end
testoutput(:,i)=vv;

original = data(:,len+1:len+testlen);
predict = testoutput(:,len+1:len+testlen);

lambda_max = 0.05; % largest lyapunov exponent 
t = (1:1:testlen)*tau1*lambda_max;
s = 1:1:M;

% plot
figure
subplot(3,1,1)
imagesc(t,s,original)
title('Actual')
xlim([0, 20])
subplot(3,1,2)
imagesc(t,s,predict)
title('Prediction')
xlim([0, 20])
caxis(3*[-1,1])
subplot(3,1,3)
imagesc(t,s,original - predict)
title('Error')
xlabel('$$\Lambda_{max}t$$', 'Interpreter', 'Latex')
caxis(3*[-1,1])
xlim([0, 20])
colormap('jet')
h=colorbar();
pos3=set(h,'Position', [0.93 0.11 0.025 0.8]);