import numpy as np

def run_methods(Y, methods):
    order = methods.keys()
    dims = {}
    i = 0
    for name in order:
        method = methods[name]
        q = method.n_components
        dims[name] = slice(i, i+q)
        i += q

    latent_spaces = np.empty((Y.shape[0], i))

    for name in methods:
        method = methods[name]
        try:
            _lat = method.fit_transform(Y)
            latent_spaces[:, dims[name]] = _lat
        except:
            raise
            print("Error detected in running method, ignoring this method as NAN")
    latent_spaces -= latent_spaces.mean(0)
    latent_spaces /= latent_spaces.std(0)
    return latent_spaces, dims
