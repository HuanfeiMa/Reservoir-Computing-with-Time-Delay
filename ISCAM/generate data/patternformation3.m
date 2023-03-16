Cs=0.5;
Cr=2.5;
p=0.565;%0.2;

N=100;%20;
L=4000;
X=zeros(N,N,L);
X(:,:,1)=2*rand(N,N);
Y=zeros(N,N,L);
Y(:,:,1)=rand(N,N);

for i=2:L
    X(:,:,i)=X(:,:,i-1)-Y(:,:,i-1).*X(:,:,i-1)+Cs;
    Ytemp=(1-p)*Y(:,:,i-1)+p*average8(Y(:,:,i-1));
    Y(:,:,i)=1./(1+exp(-5*Ytemp.*X(:,:,i-1)+Cr));
end

imagesc(Y(:,:,end));
figure
imagesc(X(:,:,end));

save('data.mat','L','N','X','Y')
