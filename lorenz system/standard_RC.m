clear all;clc
load('lorenzdata');

% data normalization,means to 0 and deviations to 1
[Data, ps] = mapstd(data);

arhow_r =0.67; % spectral radius
d=0.05; % sparsity
resSize = 200;
k=round(d*resSize);
inSize = 3; outSize = 3;
gamma = 0.44; % leaky rate
sigma = 0.44;

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
Wres2 = adj2.*Wres1;
SR = max(abs(eig(Wres2)));
Wres = Wres2 .* ( arhow_r/SR);  


initialen = 1000;
trainlen = 6000;
len = initialen+trainlen;
testlen = 3000;

r = zeros(resSize,1);
rtotal = zeros(resSize,len);
%training period
for i = 1:len
    ut = Data(:,i);
    r = (1-gamma)*r + gamma*(tanh( Win*ut + Wres*r));
    rtotal(:,i) = r;
end

rtotal = rtotal(:,initialen:len-1);
rtrain = rtotal; 
rtrain(2:2:end,:)=rtrain(2:2:end,:).^2; % half neurons are nonlinear(even terms)

% Tikhonov regularization to solve Wout
traindata = Data(:,initialen+1:len);    
beta = 1e-5; % regularization parameter
Wout = ((rtrain*rtrain' + beta*eye(resSize)) \ (rtrain*traindata'))';
trainoutput = Wout*rtrain;
mse1=mean(mean((trainoutput-traindata).^2,2));

%testing period
vv = trainoutput(:,trainlen);
testoutput = [ ];
for i = 1 : testlen
    ut = vv ; 
    r = (1-gamma)*r + gamma*(tanh( Win*ut + Wres*r));
    r2 = r;
    r2(2:2:end) = r2(2:2:end).^2;
    vv = Wout * r2;
    testoutput = [testoutput vv];
end
testdata = Data(:,len+1:len+testlen);

% plot
% 0.906--largest lyapunov exponent
t = (1:1:testlen)*0.01*0.906;
figure
subplot(3,1,1)
plot(t(1:1000),testdata(1,1:1000),'b','linewidth',1);
hold on
plot(t(1:1000),testoutput(1,1:1000),'r','linewidth',1);
xlim([0 9.06])
ylabel('x');
title('Lorenz system prediction');
subplot(3,1,2)
plot(t(1:1000),testdata(2,1:1000),'b','linewidth',1);
hold on
plot(t(1:1000),testoutput(2,1:1000),'r','linewidth',1);
ylabel('y');
xlim([0 9.06])
subplot(3,1,3)
plot(t(1:1000),testdata(3,1:1000),'b','linewidth',1);
hold on
plot(t(1:1000),testoutput(3,1:1000),'r','linewidth',1);
xlabel('$$\Lambda_{max}t$$', 'Interpreter', 'Latex')
ylabel('z');
legend('true data','RC')
xlim([0 9.06])