clear all
clc
load('data');
X=reshape(X,N*N,L);
Y=reshape(Y,N*N,L);


data=[X;Y];

% data normalization,means to 0 and deviations to 1
[Data, ps] = mapstd(data);

% reservoir parameter
resSize=1000;
inSize=size(data,1);outSize=size(data,1);
dimension=5;
d=0.05; % sparsity
gamma = 0.9; % leaky rate
tau=5;
sigma=0.01;
k=round(d*resSize);
arhow_r =0.9; % spectral radius


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
    num = randperm(resSize,k);
    for j = 1:k
        adj2(i,num(j)) = 1;
    end
end
Wres1 = -1 + 2*rand(resSize,resSize); 
Wres2 = adj2.*Wres1 ;
SR = max(abs(eig(Wres2))) ;
Wres = Wres2 .* ( arhow_r/SR);    




initialen = 200;
trainlen = 2000;
% trainlen = 2000+tau*dimension;
len = initialen+trainlen;
testlen = 1000;

r = zeros(resSize,len);
rtotal=zeros(resSize,len);
%training period
for i = 2:len
    ut = Data(:,i);
    r(:,i) = (1-gamma)*r(:,i-1) + gamma*(tanh( Win*ut + Wres*r(:,i-1)));
    rtotal(:,i) = r(:,i);
end

rtotal = rtotal(:,initialen:len-1);
traindata = Data(:,initialen+1:len);
rtrain=zeros(dimension*resSize,length(rtotal)-tau*dimension+1);

%neurons with lags
for k=1:resSize
    for i=1:dimension
        rtrain(i+dimension*(k-1),:)=rtotal(k,i*tau:end-dimension*tau+i*tau);
    end
end

rrtrain=rtrain;
rrtrain(2:2:end,:)=rrtrain(2:2:end,:).^2; % half neurons are nonlinear(even terms)   
   

% Tikhonov regularization to solve Wout
traindata=traindata(:,tau*dimension:end);
beta = 1e-6; % regularization parameter
netsize=size(rrtrain,1);
Wout = ((rrtrain*rrtrain' + beta*eye(netsize)) \ (rrtrain*traindata(:,:)'))';
mse1=mean(sum((Wout*rrtrain-traindata).^2));



r2=zeros(1,resSize*dimension);
for k=1:resSize
    for i=1:dimension
        r2(i+dimension*(k-1))=r(k,end-dimension*tau+i*tau);
    end
end
r2(2:2:end)=r2(2:2:end).^2;

%testing period
vv =Wout*r2'; 
testoutput = [];
for i = len+1 : len+testlen
    ut = vv ; 
    testoutput(:,i)=vv;
    r(:,i) = (1-gamma)*r(:,i-1) + gamma*(tanh( Win*ut + Wres*r(:,i-1)));
    
    for k=1:resSize
        for j=1:dimension
            r2(j+dimension*(k-1))=r(k,end-dimension*tau+j*tau);
        end
    end
    r2(2:2:end)=r2(2:2:end).^2;  
    vv = Wout * r2';
        
end
testoutput(:,i)=vv;


original = Data(:,len+1:len+testlen);
predict = testoutput(:,len+1:len+testlen);


%% plot
% plot X
t=1:200;
s=1:N*N;
figure
subplot(3,1,1)
imagesc(t,s,original(1:N*N,1:200))
subplot(3,1,2)
imagesc(t,s,predict(1:N*N,1:200))
subplot(3,1,3)
imagesc(t,s,original(1:N*N,1:200)-predict(1:N*N,1:200))

originalX=reshape(original(1:N*N,:),[N,N,testlen]);
predictX=reshape(predict(1:N*N,:),[N,N,testlen]);
figure
imagesc(originalX(:,:,20))
figure
imagesc(predictX(:,:,20))


% plot Y
t=1:200;
s=1:N*N;
figure
subplot(3,1,1)
imagesc(t,s,original(N*N+1:end,1:200))
subplot(3,1,2)
imagesc(t,s,predict(N*N+1:end,1:200))
subplot(3,1,3)
imagesc(t,s,original(N*N+1:end,1:200)-predict(N*N+1:end,1:200))

originalY=reshape(original(N*N+1:end,:),[N,N,testlen]);
predictY=reshape(predict(N*N+1:end,:),[N,N,testlen]);
figure
imagesc(originalY(:,:,20))
figure
imagesc(predictY(:,:,20))


% moive
h = figure;
Z = predictX(:,:,1);
imagesc(Z)
ax = gca;
ax.NextPlot = 'replaceChildren';
loops = 10;
M(loops) = struct('cdata',[],'colormap',[]);
for j = 1:loops
    X = predictX(:,:,j);
    imagesc(X)
    drawnow
    M(j) = getframe;
end
movie(M,1,2); 
