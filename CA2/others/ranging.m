close all;
ts=0.01;
fs=1/ts;

t=0:ts:10;
tlen=length(t);

x=zeros(1,tlen);
N=100;
x(1:N)=ones(1,N);

plot(t,x,'LineWidth',4)
ylim([0 1.5])
hold on

y=zeros(1,tlen);
idx1=501;
y(idx1:idx1+N-1)=0.2*ones(1,N);
idx2=801;
y(idx2:idx2+N-1)=0.1*ones(1,N);

sigma=0.05;
noise=sigma*randn(1,tlen);
y=y+noise;


plot(t,y,'LineWidth',4,'Color','r')
hold off

% First Strategy
thr=0.1;
indSig=find(y>thr);
td1=(indSig(1)-1)*ts;
C=3e8;
R1=C*td1/2;

% Second Strategy
s=ones(1,N);
ro=zeros(1,tlen-N);

for i=1:tlen-N
    ro(i)=innerproduct(y(i:i+N-1),s);
end
figure
plot(t(1:tlen-N),ro,'LineWidth',5)
xlabel('t')
[val,ind]=max(ro);
THR=10;
[pks,locs]=findpeaks(abs(ro));
indFinal=find(pks>THR);
tdhat=(locs(indFinal)-1)*ts;



indthr=find(ro>THR);






if val>THR
    
    
    
    
    
td2=(ind-1)*ts;
R2=C*td2/2;
else
   disp('No Target!') 
end

% [td1 td2]













