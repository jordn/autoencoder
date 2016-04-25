% End pad a row vector with zeros/[val] to a given length
function padded = pad(vec, len, val, dim)
if length(vec) > len
    error('Vector already too long')
end
if nargin < 3
    val = 'zeros';
end
if nargin < 4
    dim = 1;
end

if strcmp(val, 'nan')
    padvec = nan(len-length(vec));
elseif strcmp(val, 'zeros')
    padvec = zeros(len-length(vec));
end

if dim == 1
    padded = [vec; padvec];
elseif dim == 2
    padded = [vec padvec];
end
