function [seg_im] = gmm_em(file_name, k)

im = imread(file_name);
im = im2double(im);
% im = rgb2gray(im);

[n_row, n_col] = size(im);
seg_im = zeros(n_row, n_col);

init_val = zeros(k, 1);

% init with random pixels
for i = 1 : k
   init_row = uint8(rand(1) * (n_row - 1)) + 1;
   init_col = uint8(rand(1) * (n_col - 1)) + 1;
   init_val(i, 1) = im(init_row, init_col);
end

mean = init_val;
var = ones(k, 1);
post = ones(k, 1) ./ k;

latent = zeros(n_row, n_col, k);

log_likelihood = LogLikelihood(im, mean, var, post, n_row, n_col, k)

count = 1;
while 1
    % E step
    for n_r = 1 : n_row
        for n_c = 1 : n_col
            denominator = 0.0;
            for i = 1 : k
                latent(n_r, n_c, i) = post(i) * GM(im(n_r, n_c), mean(i), var(i));
                denominator = denominator + latent(n_r, n_c, i);
            end
            for i = 1 : k
                latent(n_r, n_c, i) = latent(n_r, n_c, i) / denominator;
            end
        end
    end
    % M step
    for i = 1 : k
        % update mean, var, and post
        new_mean_ele = 0.0;
        new_var_ele = 0.0;
        new_post_ele = 0.0;
        n_k = 0.0;
        for n_r = 1 : n_row
            for n_c = 1 : n_col
                n_k = n_k + latent(n_r, n_c, i);
                new_mean_ele = new_mean_ele + latent(n_r, n_c, i) * im(n_r, n_c);
                new_var_ele = new_var_ele + latent(n_r, n_c, i) * (im(n_r, n_c) - mean(i))^2;
            end
        end
        mean(i) = new_mean_ele / n_k;
        var(i) = new_var_ele / n_k;
        post(i) = n_k / (n_row * n_col);
    end
    
    new_log_likelihood = LogLikelihood(im, mean, var, post, n_row, n_col, k);
    fprintf('round %d, dif ll: %f\n', count, abs(new_log_likelihood - log_likelihood));
    if abs(new_log_likelihood - log_likelihood) < 100
        break
    else
        log_likelihood = new_log_likelihood;
    end
    count = count + 1;
end

for n_r = 1 : n_row
    for n_c = 1 : n_col
        [~, ind] = max(latent(n_r, n_c, :));
        seg_im(n_r, n_c) = mean(ind);
    end
end

end

function res = LogLikelihood(im, mean, var, post, n_row, n_col, k)

res = 0.0;
for n_r = 1 : n_row
    for n_c = 1 : n_col
        ll = 0.0;
        for i = 1 : k
            tmp = post(i, 1) * GM(im(n_r, n_c), mean(i), var(i));
            ll = ll + tmp;
        end
        ll = log(ll);
        res = res + ll;
    end
end

end

function res = GM(x, mean, var)

res = 1.0 / (var)^0.5;
res = res * exp(-(x - mean)^2 / (2 * var));

end