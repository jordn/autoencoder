% Binary value with probability given by sigmoid function
function y = sigmoidbin(z)
    y = 1 ./ (1 + exp(-z)) > rand(size(z));
end