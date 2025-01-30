function avg_error = compute_average_normalized_error(TestSignal, reconstructed)
    % Ensure dimensions match
    assert(all(size(TestSignal) == size(reconstructed)), ...
           'TestSignal and reconstructed must have the same dimensions.');

    % Compute differences and norms
    differences = TestSignal - reconstructed;          % Compute all differences
    squared_errors = vecnorm(differences, 2, 1).^2;    % L2 norm squared for differences
    signal_norms = vecnorm(TestSignal, 2, 1).^2;       % L2 norm squared for TestSignal

    % Handle division by zero
    normalized_errors = squared_errors ./ signal_norms; % Compute normalized errors
    normalized_errors(signal_norms == 0) = 0;          % Set error to 0 where signal norm is 0

    % Compute the average normalized error
    avg_error = mean(normalized_errors);
end
