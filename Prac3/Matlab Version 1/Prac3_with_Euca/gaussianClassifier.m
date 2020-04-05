function [trainAcc, testAcc] = gaussianClassifier(X, K, discriminant, covariance)
%Determine number of dimensions
d = length(X(1, :));

%If you want to randomise your training vs test datasets, this is a lazy
%way to do it. There are functions that will do this better.

trainIndices = rand(length(X(:,1)),1);
trainIndices = round(trainIndices);
trainIndices = logical(trainIndices);
testIndices = ~trainIndices;
trainX = X(trainIndices,:);
testX = X(testIndices,:);
trainK = K(trainIndices);
testK = K(testIndices);

%Define training and test values
%trainX = X(1:500,:);
%testX = X(500:end,:);


%Define training and test classes
%trainK = K(1:500);
%testK = K(500:end);


%Turn classes into logical indices
trainK1 = trainK == "pos";
trainK2 = trainK == "neg";

%Separate training X values into classes
XK1 = trainX(trainK1, :);
XK2 = trainX(trainK2, :);

%Calculate means of each class
MK1 = mean(XK1);
MK2 = mean(XK2);

%Calculate covariance matrices of each class
SK1 = cov(XK1);
SK2 = cov(XK2);

%Calculate priors
PK1 = length(trainK1)/length(trainK);
PK2 = length(trainK2)/length(trainK);

%Calculate multivariate normal probability density functions
switch discriminant
    case "quadratic"
        %Calculating probability density function for training data is only
        %useful for comparing accuracies. This is not actually used by the
        %model.
        trainLK1 = mvnpdf(trainX, MK1, SK1);
        trainLK2 = mvnpdf(trainX, MK2, SK2);
        
        %Using the means and covariance matrices from the training data,
        %evaluate gaussian probability distribution function at each row
        %of training data.
        testLK1 = mvnpdf(testX, MK1, SK1);
        testLK2 = mvnpdf(testX, MK2, SK2);
    
    %If we want a linear discriminant discriminant, we must first simplify
    %our covariances
    case "linear"
        switch covariance
            case "full"
                %Create a shared covariance matrix, weighted by priors
                sharedS = SK1*PK1 + SK2*PK2;
                
                trainLK1 = mvnpdf(trainX, MK1, sharedS);
                trainLK2 = mvnpdf(trainX, MK2, sharedS);
                testLK1 = mvnpdf(testX, MK1, sharedS);
                testLK2 = mvnpdf(testX, MK2, sharedS);
            case "diagonal"
                %Make it diagonal (all other values = zero)
                sharedS = eye(d,d).*(SK1*PK1 + SK2*PK2);
        
                trainLK1 = mvnpdf(trainX, MK1, sharedS);
                trainLK2 = mvnpdf(trainX, MK2, sharedS);
                testLK1 = mvnpdf(testX, MK1, sharedS);
                testLK2 = mvnpdf(testX, MK2, sharedS);
            
            case "same"
                %Use identity matrix, all the same, all diagonal. The
                %magnitude of this matrix won't effect where the
                %discriminant should be, but if it's too small many of the
                %probability values will be so small than MATLAB turns them
                %into logical zeros. Multiply by ~1000 in order to avoid
                %this problem.
                %If you calculate the discriminant and posterior
                %geometrically using only the means this won't be a
                %problem.
                trainLK1 = mvnpdf(trainX, MK1, 1000.*eye(d));
                trainLK2 = mvnpdf(trainX, MK2, 1000.*eye(d));
                testLK1 = mvnpdf(testX, MK1, 1000.*eye(d));
                testLK2 = mvnpdf(testX, MK2, 1000.*eye(d));
        end
end

%Calculate the numerators of the posteriors for each class.
trainNum1 = trainLK1 .* PK1;
trainNum2 = trainLK2 .* PK2;
testNum1 = testLK1 .* PK1;
testNum2 = testLK2 .* PK2;

%Calculate the full posteriors for each class.
trainPost1 = trainNum1 ./ (trainNum1 + trainNum2);
trainPost2 = trainNum2 ./ (trainNum1 + trainNum2);
testPost1 = testNum1 ./ (testNum1 + testNum2);
testPost2 = testNum2 ./ (testNum1 + testNum2);

%Initialise an array of strings for the categories.
trainCat = strings(length(trainX), 1);
testCat = strings(length(testX), 1);

%Evaluate the points based on the posterior values.
trainCat(trainPost1 >= trainPost2) = "pos";
trainCat(trainPost2 > trainPost1) = "neg";
testCat(testPost1 >= testPost2) = "pos";
testCat(testPost2 > testPost1) = "neg";

%Turn string values into categorical values so they can be compared with
%true classifications.
trainOut = categorical(trainCat);
testOut = categorical(testCat);

%Initialise array of zeros for the error of each row (0 = error, 1 =
%correct classification).
trainVal = zeros(length(trainX),1);
testVal = zeros(length(testX),1);

%Compare modelled classification with actual classification.
trainVal(trainOut == trainK) = 1;
testVal(testOut == testK) = 1;

%Calculate percentage of correct classifications.
trainAcc = 0;
testAcc = 0;
for q = 1:length(trainVal)
    if trainVal(q) == 1
        trainAcc = trainAcc + (1/length(trainK));
    end
end
for q = 1:length(testVal)
    if testVal(q) == 1
        testAcc = testAcc + (1/length(testVal));
    end
end
end

