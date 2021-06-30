function [G, B, timeSexpmt] = danielrobertexample(dim)
    [c, r] = exampleOptionPricing2(dim);
    tstamp = tic;
    [G, B] = expmt(c, r, 'alg', 'sexpm');
    timeSexpmt = toc(tstamp);
end
