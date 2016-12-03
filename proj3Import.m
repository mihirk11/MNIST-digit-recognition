addpath(pwd)
%ex1
function images = loadMNISTImages(filename)
  %loadMNISTImages returns a 28x28x[number of MNIST images] matrix containing
  %the raw MNIST images

  fp = fopen(filename, 'rb');
  assert(fp ~= -1, ['Could not open ', filename, '']);

  magic = fread(fp, 1, 'int32', 0, 'ieee-be');
  assert(magic == 2051, ['Bad magic number in ', filename, '']);

  numImages = fread(fp, 1, 'int32', 0, 'ieee-be');
  numRows = fread(fp, 1, 'int32', 0, 'ieee-be');
  numCols = fread(fp, 1, 'int32', 0, 'ieee-be');

  images = fread(fp, inf, 'unsigned char');
  images = reshape(images, numCols, numRows, numImages);
  %images = permute(images,[2 1 3]);
  images = permute(images,[1 2 3]);
  fclose(fp);

  % Reshape to #pixels x #examples
  images = reshape(images, size(images, 1) * size(images, 2), size(images, 3));
  % Convert to double and rescale to [0,1]
  images = double(images) / 255;

endfunction


function labels = loadMNISTLabels(filename)
  %loadMNISTLabels returns a [number of MNIST images]x1 matrix containing
  %the labels for the MNIST images

  fp = fopen(filename, 'rb');
  assert(fp ~= -1, ['Could not open ', filename, '']);

  magic = fread(fp, 1, 'int32', 0, 'ieee-be');
  assert(magic == 2049, ['Bad magic number in ', filename, '']);

  numLabels = fread(fp, 1, 'int32', 0, 'ieee-be');

  labels = fread(fp, inf, 'unsigned char');

  assert(size(labels,1) == numLabels, 'Mismatch in label count');

  fclose(fp);

endfunction


% Change the filenames if you've saved the files under different names
% On some platforms, the files might be saved as 
% train-images.idx3-ubyte / train-labels.idx1-ubyte
images = loadMNISTImages('train-images.idx3-ubyte');
labels = loadMNISTLabels('train-labels.idx1-ubyte');
 
 %imagesc(reshape(images(:,1),28,28))
% We are using display_network from the autoencoder code
%display_network(images(:,1:100)); % Show the first 100 images
%disp(labels(1:10));