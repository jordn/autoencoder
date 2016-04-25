% End pad a row vector with zeros to a given length
function padded = pad(vec, len, dim)
if length(vec) > len
    error('Vector already too long')
end
if nargin < 3
    dim = 1
end
if dim == 1
    padded = [vec; zeros(len-length(vec), 1)];
elseif dim == 2
    padded = [vec zeros(len-length(vec), 1)];
end
