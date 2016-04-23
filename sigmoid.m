function y = sigmoid(z)
    y = 1 ./ (1 + exp(-z));
end