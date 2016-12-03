% This pseudo-code illustrates implementing a several layer neural 
%network. You need to fill in the missing part to adapt the program to 
%your own use. You may have to correct minor mistakes in the program
%% prepare for the data
%load dataMatlab.mat
train_x = trainImages;
test_x = validImages;
train_y = trainLabels.';
test_y = validLabels.';
%% Some other preparations
%Number of hidden layers
numOfHiddenLayer = 1;
s{1} = size(train_x, 1);
s{2} = 5;
s{3} = 10;
%s{4} = 100;
%s{5} = 2;
%Initialize the parameters
%You may set them to zero or give them small
%random values. Since the neural network
%optimization is non-convex, your algorithm
%may get stuck in a local minimum which may
%be caused by the initial values you assigned.
 W{1} = rand(784,5);
 b{1} = rand(1,5);
 W{2} = rand(784,10);
 b{2} = rand(1,10);

losses = [];
train_errors = [];
test_wrongs = [];
%Here we perform mini-batch stochastic gradient descent
%If batchsize = 1, it would be stochastic gradient descent
%If batchsize = N, it would be basic gradient descent
batchsize = 100;
%Num of batches
numbatches = size(train_x, 2) / batchsize;
%% Training part
%Learning rate alpha
alpha = 0.01;
%Lambda is for regularization
lambda = 0.001;
%Num of iterations
numepochs = 20;
for j = 1 : numepochs
 %randomly rearrange the training data for each epoch
 %We keep the shuffled index in kk, so that the input and output could
 %be matched together
 kk = randperm(size(train_x, 2));
 for l = 1 : numbatches
 %Set the activation of the first layer to be the training data
 %while the target is training labels
 a{1} = train_x(:, kk( (l-1)*batchsize+1 : l*batchsize ) );
 y = train_y(:, kk( (l-1)*batchsize+1 : l*batchsize ) );
 %Forward propagation, layer by layer
 %Here we use sigmoid function as an example
 for i = 2 : numOfHiddenLayer + 1
 %a{i} = sigm( bsxfun(@plus, W{i-1}*a{i-1}, b{i-1}) );
 a{i} = 1./(1+exp(-( bsxfun(@plus, sum((W{i-1}.'*a{i-1}).'), b{i-1}) )));%CHANGED BY MIHIR
 end
 %Calculate the error and back-propagate error layer by layers
 d{numOfHiddenLayer + 1} = -(y - a{numOfHiddenLayer + 1}) .* a{numOfHiddenLayer + 1} .* (1-a{numOfHiddenLayer + 1});
 for i = numOfHiddenLayer : -1 : 2
 d{i} = W{i}' * d{i+1} .* a{i} .* (1-a{i});
 end

 %Calculate the gradients we need to update the parameters
 %L2 regularization is used for W
 for i = 1 : numOfHiddenLayer
 dW{i} = d{i+1} * a{i}';
 db{i} = sum(d{i+1}, 2);
 W{i} = W{i} - alpha * (dW{i} + lambda * W{i});
 b{i} = b{i} - alpha * db{i};
 end
 end 
 
  % Do some predictions to know the performance
 a{1} = test_x;
 % forward propagation
 for i = 2 : numOfHiddenLayer + 1
 %This is essentially doing W{i-1}*a{i-1}+b{i-1}, but since they
 %have different dimensionalities, this addition is not allowed in
 %matlab. Another way to do it is to use repmat
 a{i} = sigm( bsxfun(@plus, W{i-1}*a{i-1}, b{i-1}) );
 end
 %Here we calculate the sum-of-square error as loss function
 loss = sum(sum((test_y-a{numOfHiddenLayer + 1}).^2)) / size(test_x, 2);
 % Count no. of misclassifications so that we can compare it
 % with other classification methods
 % If we let max return two values, the first one represents the max
 % value and second one represents the corresponding index. Since we
 % care only about the class the model chooses, we drop the max value
 % (using ~ to take the place) and keep the index.
 [~, ind_] = max(a{numOfHiddenLayer + 1}); [~, ind] = max(test_y);
 test_wrong = sum( ind_ ~= ind ) / size(test_x, 2) * 100;

 %Calculate training error
 %minibatch size
 bs = 2000;
 % no. of mini-batches
 nb = size(train_x, 2) / bs;
 train_error = 0;
 %Here we go through all the mini-batches
 for ll = 1 : nb
 %Use submatrix to pick out mini-batches
 a{1} = train_x(:, (ll-1)*bs+1 : ll*bs );
 yy = train_y(:, (ll-1)*bs+1 : ll*bs );
 for i = 2 : numOfHiddenLayer + 1
 a{i} = sigm( bsxfun(@plus, W{i-1}*a{i-1}, b{i-1}) );
 end
 train_error = train_error + sum(sum((yy-a{numOfHiddenLayer + 1}).^2));
 end
 train_error = train_error / size(train_x, 2);
 losses = [losses loss];
 test_wrongs = [test_wrongs, test_wrong];
 train_errors = [train_errors train_error];
end 