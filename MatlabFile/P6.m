


in_data = zeros(254, 6, 1, 9);
out_data = [];

f_name = ["1/1_", "2/2_", "3/3_", "4/4_", "5/5_", "6/6_", "7/7_", "8/8_", "9/9_", "0/0_"];

for j = 1:50
    for i=1:10
        f = strcat('keyboard/', f_name(i), int2str(j), '.csv');
        t = csvread(f);
        in_data(:, :, 1, (j-1)*10+i) = t(2:255, :);
        z = eye(10);
        out_data  = [out_data; z(i, :)];
    end
end

net = [
    imageInputLayer([254, 6])
    convolution2dLayer([2 2],10,'Stride',2)
    batchNormalizationLayer
    reluLayer
    %averagePooling2dLayer(3,'Stride',3)
    dropoutLayer(0.5)
    fullyConnectedLayer(200)
    fullyConnectedLayer(10)
    regressionLayer
    ];

%regressionLayer

options = trainingOptions('sgdm',...
    'MiniBatchSize',50,...
    'MaxEpochs',50,...
    'InitialLearnRate',1e-2,...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropFactor',0.5,...
    'LearnRateDropPeriod',10,...
    'Shuffle','every-epoch',...
    'Plots','training-progress',...
    'Verbose',false);

net_result = trainNetwork(in_data(:, :, :, 1:250), out_data(1:250, :), net, options);

hit = 0;
nothit = 0;

con_mat = zeros(10, 10);

for i = 1:250
    this_data = in_data(:, :, 1, i);
    [a, label] = max(out_data(i, :));
    otest = predict(net_result, this_data);
    [b, out] = max(otest);
    if out == label
        hit = hit+1;
        fprintf("test%d hit %d\n", i, out);
    else
        nothit = nothit+1;
        %fprintf("test%d nothit\n", i);
    end
    con_mat(label, out) = con_mat(label, out)+1;
end

heatmap([1:9, 0], [1:9, 0], con_mat);

hit = 0;
nothit = 0;

con_mat = zeros(10, 10);

for i = 251:500
    this_data = in_data(:, :, 1, i);
    [a, label] = max(out_data(i, :));
    otest = predict(net_result, this_data);
    [b, out] = max(otest);
    if out == label
        hit = hit+1;
        fprintf("test%d hit %d\n", i, out);
    else
        nothit = nothit+1;
        %fprintf("test%d nothit\n", i);
    end
    con_mat(label, out) = con_mat(label, out)+1;
end
figure();
heatmap([1:9, 0], [1:9, 0], con_mat);

