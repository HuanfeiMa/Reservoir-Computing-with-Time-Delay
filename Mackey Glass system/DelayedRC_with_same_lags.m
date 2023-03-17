clear all;clc
load('MGdata');
% data normalization,means to 0 and deviations to 1
[Data, ps] = mapstd(data');

resSize=50;
inSize=1;outSize=1;
dimension=5; % same lags for each neuron
gamma = 0.44; % leaky rate
tau=5;
sigma=0.44;
d = 0.2; % sparsity
q=round(d*resSize);
arhow_r =0.79; % spectral radius

% generate weight matrix
Win1 = -1 + 2*rand(resSize,inSize);
adj1 = zeros(resSize,inSize);
for m=1:resSize
    for n=1:inSize
        if(rand(1,1)<sigma)  
            adj1(m,n)=1;  
        end
    end
end
Win = adj1.*Win1;


adj2 = zeros(resSize,resSize);
for i = 1:resSize
    num = randperm(resSize,q);
    for j = 1:q
        adj2(i,num(j)) = 1;
    end
end
Wres1 = -1 + 2*rand(resSize,resSize); 
Wres2 = adj2.*Wres1 ;
SR = max(abs(eig(Wres2))) ;
Wres = Wres2 .* ( arhow_r/SR);            

initialen = 10000;
trainlen = 25000;
% trainlen = 25000+tau*dimension;
len = initialen+trainlen;
testlen = 6000;
r = zeros(resSize,len);

%training period
rtotal=zeros(resSize,len);
for i = 2:len
    ut = Data(i);
    r(:,i) = (1-gamma)*r(:,i-1) + gamma*(tanh( Win*ut + Wres*r(:,i-1)));
    rtotal(:,i) = r(:,i);
end
rtotal = rtotal(:,initialen:len-1);
traindata = Data(initialen+1:len);
rtrain=[];

%neurons with lags
for k=1:resSize
    for i=1:dimension
        rtrain(i+dimension*(k-1),:)=rtotal(k,i*tau:end-dimension*tau+i*tau); %time: len-1
    end
end
rrtrain=rtrain;
rrtrain(2:2:end,:)=rtrain(2:2:end,:).^2; % half neurons are nonlinear(even terms)   

% Tikhonov regularization to solve Wout
traindata=traindata(tau*dimension:end);
beta = 1e-4; % regularization parameter
netsize=size(rrtrain,1);
Wout = ((rrtrain*rrtrain' + beta*eye(netsize)) \ (rrtrain*traindata(:,:)'))';
mse1=mean((Wout*rrtrain-traindata).^2,2);

for k=1:resSize
    for i=1:dimension
        r2(i+dimension*(k-1))=r(k,end-dimension*tau+i*tau);
    end
end
r2(2:2:end)=r2(2:2:end).^2;

%testing period
vv =Wout*r2';
testoutput = zeros(3,testlen);
rr=r(end);
for i = len+1 : len+testlen
    ut = vv ; 
    testoutput(:,i)=vv;
    r(:,i) = (1-gamma)*r(:,i-1) + gamma*(tanh( Win*ut + Wres*r(:,i-1)));
    for k=1:resSize
        for i=1:dimension
            r2(i+dimension*(k-1))=r(k,end-dimension*tau+i*tau);
        end
    end
    r2(2:2:end)=r2(2:2:end).^2;
    vv = Wout * r2';   
end
testoutput(:,i)=vv;

original = Data(:,len+1:len+testlen);
predict = testoutput(:,len+1:len+testlen);

% plot
% 0.006--largest lyapunov exponent
t = (1:1:testlen)*0.5*0.006;
figure
plot(t(1:1000),original(1,1:1000),'b','linewidth',1);
hold on
plot(t(1:1000),predict(1,1:1000),'r','linewidth',1);
title('MG system prediction')
xlabel('$$\Lambda_{max}t$$', 'Interpreter', 'Latex')
ylabel('x');
legend('true data','Delayed RC')