load('proj3.mat');

train_x = trainImages;% 784*60k
test_x = validImages;% 784*10k
train_y = trainLabels.';%1*60k
test_y = validLabels.';%1*10k

J=100;
K=10;
D=784;
eta=0.05;
count=0;

trainSize=size(train_y);
testSize=size(test_y);
t=zeros(trainSize(2),10);

W{1} = (rand(D,J)*2)-1;
%b{1} = (rand(1,J)-0.5)*10;
b{1} = 1;
W{2} = (rand(J,K)*2)-1;
%b{2} = (rand(1,K)-0.5)*10;
b{2} = 1;

for i=1:trainSize(2)
  t(i,train_y(i)+1)=1;
end

numberOfIterations=10;
for k2=1:numberOfIterations
    for i2=1:trainSize(2)
        
        %z=1./(1+exp(-((W{1}.'*train_x(:,i2)).')+b{1}));
        z2=(W{1}.'*train_x(:,i2)).'+b{1};
        %z=arrayfun(@(x1) (1/1+exp(-1*x1)),z2);
        z=tanh(z2);
        
        a=(W{2}.'*z.').'+b{2};
        
        %y=exp(a)/sum(exp(a));
        sum=0;
        for i4=1:K
            sum=sum+exp(a(1,i4));
        end
        for i4=1:K
            y(1,i4)=exp(a(1,i4))/sum;
        end
        
        dkWhole=y-t(i2,:);
        %dkWhole=dkWhole.*(exp(y)./((1+exp(y)).*(1+exp(y))));
        %sigmDervz=exp(z)./((1+exp(z)).*(1+exp(z)));
        %djWhole=(exp(z)./((1+exp(z)).*(1+exp(z)))).*((W{2}*dkWhole.').');
        dja=(sech(z)).^2;
        djb=((W{2}*dkWhole.').');
        djWhole=dja.*djb;
        %djWhole=(-exp(-z)./((1+exp(-z)).*(1+exp(-z)))).*((W{2}*dkWhole.').');
        %djWhole=dot((exp(z)./((1+exp(z)).*(1+exp(z)))),((W{2}*dkWhole.').'));
        dE1=djWhole.'*train_x(:,i2).';
        dE2=dkWhole.'*z;

        W{1}=W{1}-eta*dE1.';
        W{2}=W{2}-eta*dE2.';
    end
    eta=eta/2;
end
maxY=zeros(1,testSize(2));
for i2=1:testSize(2)
        %z=1./(1+exp(-((W{1}.'*train_x(:,i2)).')+b{1}));
        z2=(W{1}.'*train_x(:,i2)).'+b{1};
        %z=arrayfun(@(x1) (1/1+exp(-1*x1)),z2);
        z=tanh(z2);
        
        a=(W{2}.'*z.').'+b{2};
        %y=exp(a)/sum(exp(a));
        sum=0;
        for i4=1:K
            sum=sum+exp(a(1,i4));
        end
        for i4=1:K
            y(1,i4)=exp(a(1,i4))/sum;
        end
        maxY(1,i2)=find(y == max(y), 1)-1;
        if(find(y == max(y), 1)==test_y(i2)+1)
            count=count+1;
        end
end
count/testSize(2)

b{1} = ones(1,J);
b{2} = ones(1,K);

Wnn1=W{1};
bnn1=b{1};
Wnn2=W{2};
bnn2=b{2};
h='tanh';
save('proj3.mat','Wlr','blr','Wnn1','Wnn2','bnn1','bnn2','h'); 
%save('proj3.mat','Wnn1','Wnn2','bnn1','bnn2','h');