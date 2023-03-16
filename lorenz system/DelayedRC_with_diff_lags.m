clear all;clc
load('lorenzdata');

% data normalization,means to 0 and deviations to 1
[Data, ps] = mapstd(data);

resSize=40;
inSize=3;outSize=3;

% different lags for each neuron
% keep the total dimension=200
mu=5;
dq=round(normrnd(mu,0.5,[1 resSize]));
dq(resSize/2+1:end)=2*mu-dq(1:resSize/2);
dimension=sum(dq);

gamma = 0.44; % leaky rate
tau=5;
sigma=0.44;
d=0.05; % sparsity
k=round(d*resSize);
arhow_r =0.67; % spectral radius

% generate weight matrix
Win1 = -0.5 + rand(resSize,inSize);
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
    num = randperm(resSize,k);
    for j = 1:k
        adj2(i,num(j)) = 1;
    end
end
Wres1 = -1 + 2*rand(resSize,resSize); 
Wres2 = adj2.*Wres1 ;
SR = max(abs(eig(Wres2))) ;
Wres = Wres2 .* ( arhow_r/SR);    


initialen = 1000;
trainlen = 6000;
% trainlen = 6000+tau*dimension;
len = initialen+trainlen;
testlen = 3000;
r = zeros(resSize,len);

%training period
rtotal=zeros(resSize,len);
for i = 2:len
    ut = Data(:,i);
    r(:,i) = (1-gamma)*r(:,i-1) + gamma*(tanh( Win*ut + Wres*r(:,i-1)));
    rtotal(:,i) = r(:,i);
end
rtotal = rtotal(:,initialen:len-1);
rtrain=[];

%neurons with different lags
for k=1:resSize
    if k==1
        rows=0;
    else
        rows=sum(dq(1:k-1));
    end
    for i=1:dq(k)
        rtrain(i+rows,:)=rtotal(k,i*tau:end-max(dq)*tau+i*tau);
    end
end
rrtrain=rtrain;
rrtrain(2:2:end,:)=rtrain(2:2:end,:).^2; % half neurons are nonlinear(even terms)

 % Tikhonov regularization to solve Wout   
traindata = Data(:,initialen+1:len);
traindata=traindata(:,tau*max(dq):end);
beta = 1e-5; % regularization parameter
netsize=size(rrtrain,1);
Wout = ((rrtrain*rrtrain' + beta*eye(netsize)) \ (rrtrain*traindata(:,:)'))';
trainoutput=Wout*rrtrain;
mse1=mean(sum((trainoutput-traindata).^2));

for k=1:resSize
    if k==1
        rows=0;
    else
        rows=sum(dq(1:k-1));
    end
    for i=1:dq(k)
        r2(i+rows,:)=r(k,end-max(dq)*tau+i*tau);
    end
end
r2(2:2:end) = r2(2:2:end).^2;

%testing period
vv =Wout*r2;
testoutput = zeros(3,testlen);
for i = len+1 : len+testlen
    ut = vv ; 
    testoutput(:,i)=vv;
    r(:,i) = (1-gamma)*r(:,i-1) + gamma*(tanh( Win*ut + Wres*r(:,i-1)));
    for k=1:resSize
        if k==1
            rows=0;
        else
            rows=sum(dq(1:k-1));
        end
        for j=1:dq(k)
            r2(j+rows)=r(k,end-max(dq)*tau+j*tau);
        end
    end
    r2(2:2:end) = r2(2:2:end).^2;
    vv = Wout * r2;    
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