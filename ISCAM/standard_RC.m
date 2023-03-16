clear all;clc
load('data');
X=reshape(X,N*N,L);
Y=reshape(X,N*N,L);

data=[X;Y];

% data normalization,means to 0 and deviations to 1
[Data, ps] = mapstd(data);


% reservoir parameter
arhow_r =0.9; % spectral radius
d=0.05; % sparsity
resSize = 5000;
k=round(d*resSize);
inSize = size(data,1); outSize = size(data,1);
gamma = 0.9; % leaky rate
sigma = 0.01;


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



initialen = 200;
trainlen = 2000;
len = initialen+trainlen;
testlen = 1000;


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
rtrain(2:2:end,:)=rtotal(2:2:end,:).^2; % half neurons are nonlinear(even terms)   


% Tikhonov regularization to solve Wout
traindata = Data(:,initialen+1:len);    
beta = 1e-6; % regularization parameter
Wout = ((rtrain*rtrain' + beta*eye(resSize)) \ (rtrain*traindata'))';
trainoutput = Wout*rtrain;    
mse1=mean(sum((trainoutput-traindata).^2));


r=rtrain(:,end);
%testing period
vv = trainoutput(:,end);
testoutput = [ ];
for i = 1 : testlen
    ut = vv ; 
    r = (1-gamma)*r + gamma*(tanh( Win*ut + Wres*r));
    r2 = r;
    r2(2:2:end)=r2(2:2:end).^2;
    vv = Wout * r2;
    testoutput = [testoutput vv];
end

testdata = Data(:,len+1:len+testlen);

%% plot
% plot X
t=1:200;
s=1:N*N;
figure
subplot(3,1,1)
imagesc(t,s,testdata(1:N*N,1:200))
subplot(3,1,2)
imagesc(t,s,testoutput(1:N*N,1:200))
subplot(3,1,3)
imagesc(t,s,testdata(1:N*N,1:200)-testoutput(1:N*N,1:200))

originalX=reshape(testdata(1:N*N,:),[N,N,testlen]);
predictX=reshape(testoutput(1:N*N,:),[N,N,testlen]);
figure
imagesc(originalX(:,:,20))
figure
imagesc(predictX(:,:,20))


% plot Y
t=1:200;
s=1:N*N;
figure
subplot(3,1,1)
imagesc(t,s,testdata(N*N+1:end,1:200))
subplot(3,1,2)
imagesc(t,s,testoutput(N*N+1:end,1:200))
subplot(3,1,3)
imagesc(t,s,testdata(N*N+1:end,1:200)-testoutput(N*N+1:end,1:200))

originalY=reshape(testdata(N*N+1:end,:),[N,N,testlen]);
predictY=reshape(testoutput(N*N+1:end,:),[N,N,testlen]);
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
