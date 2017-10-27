function [ cM ] = calcConfusionMatrix( Lclass, Ltrue )
classes = unique(Ltrue);
numClasses = length(classes);
cM = zeros(numClasses);

for i = 1:numClasses
    for j = 1:numClasses
        cM(i,j) = sum(Lclass(Ltrue == classes(i)) == classes(j));
    end
end
end

