
% Collecting results for training size analysis (if needed)
%results_N = zeros(60, 3);

%% Looping through training sample size (if needed)
for i = 200
    i
    
    %% Generating training subsets
    %trainSubset1 = datasample(train, i*5000);
    %trainSubset2 = datasample(train, i*5000);
    trainSubset1 = train(1:1000000, :);
    trainSubset2 = datasample(train(1000000:end, :), 100000);

    % processing subsets to remove broken data points and interpret information
    %  into a higher-dimensional vector (currently R18)
    [trainData1, trainFare1, labels1] = processTrainingData(trainSubset1);
    [trainData2, trainFare2, labels2] = processTrainingData(trainSubset2);

    % interpretting testing data WITHOUT attempting to remove broken points,
    %  for Kaggle evaluation purposes
    testData = processTestData(test);
    
    % Collecting results for ensemble size analysis (if needed)
    %results_S = zeros(50, 3);
    
    %% Looping through ensemble sizes (if needed)
    for j = 10
        j
        
        %% Modeling and predicting
        % generating  a Tree Regression Ensemble model using Bagging
        %  (LS Boosting consistently led to overfitting)
        mdl = fitrensemble(trainData1, trainFare1, 'Method', 'Bag', 'NumLearningCycles', j*10, 'NPrint', 1);

        fares1 = predict(mdl, trainData1);
        fares2 = predict(mdl, trainData2);
        faresTest = predict(mdl, testData);

        diff1 = abs(fares1 - trainFare1);
        diff2 = abs(fares2 - trainFare2);
        aveDiff1 = mean(diff1);
        aveDiff2 = mean(diff2);
        rms1 = sqrt(mean(diff1.^2));
        rms2 = sqrt(mean(diff2.^2));
        
        % Adding results for emsemble size analysis
        %results_S(j, :) = [j*10 rms1 rms2];

    end
    
    % Adding results for training size analysis
    %results_N(i, :) = [i*5000-2500 rms1 rms2];
    
    %results = [test{:, 1} faresTest];
    %writetable(array2table(results, 'VariableNames', {'key' 'fare_amount'}), 'Results.csv');

    %% Messing around with predictor importance
    
    imp = predictorImportance(mdl)/sum(predictorImportance(mdl));
    figure(1)
    bar(imp(2:end))
    set(gca, 'XTick', 1:length(labels1)-1, 'XTickLabel', labels1(2:end));
    %set(gca, 'xticklabel', labels1(2:end))
    figure(2)
    bar(imp)
    set(gca, 'XTick', 1:length(labels1), 'XTickLabel', labels1);
    %set(gca, 'xticklabel', labels1)
    
    
end


%% Data Treatment Functions
function [fareData, fare, labels] = processTrainingData(oldData)

dist = distance(oldData{:, 5}, oldData{:, 4}, ...
    oldData{:, 7}, oldData{:, 6});

margin = 0.003;
limit = 0.3;

oldData = oldData(...
    (dist > margin) & (dist < limit) & ...
    (oldData{:, 2} > 0) & (oldData{:, 8} > 0), ...
    1:end);

x1 = oldData{:, 5};
y1 = oldData{:, 4};
x2 = oldData{:, 7};
y2 = oldData{:, 6};

dist = distance(x1, y1, x2, y2);

d1p = distance(x1, y1, 40.7141667, -74.0063889);
d2p = distance(x1, y1, 40.639722, -73.778889);
d3p = distance(x1, y1, 40.6925, -74.168611);
d4p = distance(x1, y1, 40.77725, -73.872611);

d1d = distance(40.7141667, -74.0063889, x2, y2);
d2d = distance(40.639722, -73.778889, x2, y2);
d3d = distance(40.6925, -74.168611, x2, y2);
d4d = distance(40.77725, -73.872611, x2, y2);

date = datetime(extractBefore(oldData{:, 3} ,' UTC'), ...
    'InputFormat','yyyy-MM-dd HH:mm:ss');

time = date.Second + date.Minute*60 + date.Hour*3600;
dayW = weekday(date);
dayY = day(date, 'dayofyear');

fareData = [dist,  oldData{:, 7}, ...
    x1, y1, x2, y2, ...
    d1p, d2p, d3p, d4p, d1d, d2d, d3d, d4d, ...
    time, dayW, dayY, date.Year];

fare = oldData{:, 2};

labels = {"Distance" "Passengers" ...
    "LatP" "LongP" "LatD" "LongD" ...
    "NyP" "JfkP" "EwrP" "LgaP" ...
    "NyD" "JfkD" "EwrD" "LgaD" ...
    "Time" "DayOfWeek" "DayOfYear" "Year"};

end

function [fareData] = processTestData(oldData)

x1 = oldData{:, 4};
y1 = oldData{:, 3};
x2 = oldData{:, 6};
y2 = oldData{:, 5};

dist = distance(x1, y1, x2, y2);

d1p = distance(x1, y1, 40.7141667, -74.0063889);
d2p = distance(x1, y1, 40.639722, -73.778889);
d3p = distance(x1, y1, 40.6925, -74.168611);
d4p = distance(x1, y1, 40.77725, -73.872611);

d1d = distance(40.7141667, -74.0063889, x2, y2);
d2d = distance(40.639722, -73.778889, x2, y2);
d3d = distance(40.6925, -74.168611, x2, y2);
d4d = distance(40.77725, -73.872611, x2, y2);

date = datetime(extractBefore(cellstr(oldData{:, 2}) ,' UTC'), ...
    'InputFormat','yyyy-MM-dd HH:mm:ss');

time = date.Second + date.Minute*60 + date.Hour*3600;
dayW = weekday(date);
dayY = day(date, 'dayofyear');

fareData = [dist,  oldData{:, 7}, ...
    x1, y1, x2, y2, ...
    d1p, d2p, d3p, d4p, d1d, d2d, d3d, d4d, ...
    time, dayW, dayY, date.Year];

end
