
import numpy as np

def make_cell_division_times(n_divisions, n_replicates=8, std=.1, seed=None, drop_p=.05, maxtime=10):
    """
    Simulate Cell division times for n_divisions. The division times are drawn between 0 and maxtime.

    Dropout events are simulated for cells to be dropped due to technical variance.

    Cell stages increase exponentially (*2) for each cell division.

    The first cell stage cannot have dropout events.

    :param n_divisions: number of divions the cells do.
    :param n_replicates: number of cells in the beginning.
    :param std: amount of noise (in standard deviation) to add.
    :param seed: seed for reproduceability.
    :param drop_p: drop probability for each cell after first cell stage.
    :param maxtime: the end point in time.
    """
    seed = seed or np.random.randint(1000,10000)
    np.random.seed(seed)

    # Make random blobs of data with varying varainces, dependend on blob size.
    # This reflects cells of similar expression, and therefore, expresses cell stages/differentiation.
    class_sizes = n_replicates*(2**np.arange(0, n_divisions+1))#np.diff(np.r_[0, class_centers, n_data])
    n_data = class_sizes.sum()

    class_centers = (np.linspace(0,n_data,n_divisions+2)).astype(int)
    class_centers = np.int64((class_centers[:-1] + class_centers[1:])/2)

    class_vars = np.ones(class_sizes.size)*std#(max_variance/class_sizes.max())*class_sizes
    #class_vars = (std/class_sizes.max())*class_sizes
    # Each cell has their own center, such that the higher
    # cell stages have a plateue distribution (multimodal normal distribution)
    # such that biological variation is captured better
    t = []
    for mu, st, size in zip(np.exp(np.linspace(0,np.log(maxtime),n_data)[class_centers]), class_vars, class_sizes):
        stage = int(size/n_replicates)
        offset = (2*st)#/float(stage)
        # make cell centers for each cell stage:
        # cell_centers = np.random.normal(mu, var/2., stage)
        cell_centers = np.random.uniform((1-.1)*mu, (1+.1)*mu, stage)
        # Then make replicates around the centers
        cell_individuals = np.random.uniform(cell_centers-offset, cell_centers+offset, (n_replicates,stage)).flatten()
        # cell_individuals = np.random.normal(cell_centers, var, (n_replicates,stage)).flatten()
        #cell_individuals = np.random.uniform((1-offset)*mu, (1+offset)*mu, n_replicates*stage)
        t = np.r_[t, cell_individuals]
    t = t[:,None]
    t.sort(0)

    labels = []
    cell_stage = 1
    for size in class_sizes:
        labels.extend([cell_stage]*size)
        cell_stage *= 2
    labels = np.r_[labels]

    # Now we want to simulate some techincal drop out events,
    # where cells where destroyed or fell out due to technical variation,
    # Thus, we go hyper geometric, and the first cell stage cannot
    # fall out:
    #dropout_p = np.zeros(n_replicates)
    #for i in range(1,n_divisions+1):
    #    stage_size = n_replicates*(2**i)
    #    dropout_p = np.r_[dropout_p, np.repeat(2.0**i, stage_size)]
    #dropout_p /= dropout_p.max()
    #dropout_p *= drop_p

    dropout_p = np.ones(n_data) * drop_p
    dropout_p[:n_replicates] = 0

    dropouts = np.random.binomial(1, 1.-dropout_p).astype(bool)

    return t[dropouts], labels[dropouts], seed

import GPy
from GPy.util import diag
from scipy.stats import norm

def simulate_latent_space(t, labels, seed=None, var=.2, split_prob=.1, gap=.75):
    """
    Simulate splitting events in the latent space. The input time t is
    a one dimensional array having the times in it. The labels is a int
    array-like, which holds the labels for the wanted cell types.
    Basically it is an array of repetitions of 1 to number of cell types,
    e.g.: array([1..1,2..2,3..3,4..4]) for 4 cell types.

    :param array_like t: the time as [nx1] array, where n is the number of cells.
    :param array_like labels: the labels for the cells before splitting.
    :param int seed: the seed for this splitting, for reproducability.
    :param scalar var: the variance of spread of the first split, increasing after that.
    :param [0,1] split_prop: probability of split in the beginning, halfs with each split.
    :param [0,1] gap: the gap size between splitends and the beginning of the next.

    The method returns Xsim, seed, labels, time::

        - Xsim is the two dimensional latent space with splits included.
        - seed is the seed generated, for reproduceability.
        - labels are the corrected labels, for split events.
        - time is the corrected timeline for split events.
    """
    seed = seed or np.random.randint(1000,10000)
    np.random.seed(seed)

    n_data = t.shape[0]
    newlabs = []

    assert np.issubdtype(labels.dtype, np.int_) and np.greater(labels, 0).all(), "labels need to be of positive integer dtype, 0 is not allowed"

    ulabs = []
    for x in range(n_data):
        if labels[x] not in ulabs:
            ulabs.append(labels[x])

    Xsim = np.zeros((n_data, 2))
    split_ends = [Xsim[0]]
    prev_ms = [[.1,.1]]
    split_end_times = [t[labels==ulabs[0]].max()]

    t = np.sort(t.copy(), 0)

    tmax = t.max()

    for lab in ulabs:
        fil = (lab==labels).nonzero()[0]

        # zero out, for simulating linear relation within cluster:
        new_se = []
        new_m = []
        new_set = []

        splits = np.array_split(fil, len(split_ends))

        i = 1
        for s in range(len(split_ends)):
            # for all previously done splits:
            prev_m = prev_ms[s]
            split = splits[s]
            split_end = split_ends[s]
            split_end_time = split_end_times[s]

            pre_theta = None
            prev_split_time = None
            for split in np.array_split(split, np.random.binomial(1, split_prob)+1):
                newlabs.extend(["{} {}".format(_c, i) for _c in labels[split]])
                i += 1
                # If we split a collection into two, we want the two times to match up now:
                if prev_split_time is None:
                    prev_split_time = t[split].ptp()
                else:
                    t[split.min():] -= prev_split_time
                t[split] -= (t[split.min()]-split_end_time)

                # make splits longer, the farther in we are into
                # the split process, it scales with sqrt(<split#>)
                x = t[split].copy()
                x -= x.min()
                x /= x.max()
                x *= np.sqrt(lab)

                # rotate m away a little from the previous direction:
                if pre_theta is None:
                    pre_theta = theta = np.random.uniform(-45, 45)
                else:
                    theta = ((pre_theta+90)%90)-90
                theta *= (np.pi/180.) # radians for rotation matrix
                rot_m = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                m = np.dot(rot_m, prev_m)

                # later splits have bigger spread:
                v = (x.mean(0) - np.abs((-x+x.mean(0))))
                v -= v.min(0)-1e-6
                v /= v.max(0)
                v *= var*t[split]/tmax

                # make the split
                Xsim[split] = np.random.normal(split_end + m*x, v)

                # put a gap between this and the next split:
                p = m*x[-1]
                #p /= np.sqrt(GPy.util.linalg.tdot(p))

                # save the new sets of splits
                new_se.append(split_end + (1+gap)*p)
                new_m.append(m)
                new_set.append(t[split.max()])

        split_ends = new_se
        prev_ms = new_m
        split_end_times = new_set
        # The split probability goes up every time the cell stage changes:

        split_prob = min(1., split_prob*2)

    Xsim -= Xsim.mean(0)
    Xsim /= Xsim.std(0)
    #Xsim += np.random.normal(0,var,Xsim.shape)

    from scipy.stats import t as tdist
    Xsim += tdist.rvs(3, loc=0, scale=.1*var, size=Xsim.shape) #Add outliers


    return Xsim, seed, np.asarray(newlabs), t

from scipy import stats

def simulate_new_Y_RNAseq(Xsim, t, p_dims,
                          num_classes=10,
                          noise_var=.2, 
                          dropout_dist=stats.logistic(0, 1), dropout_p=.1):
    """
    Simulate an RNAseq experiment. 
    
    We model dropouts in RNA count space (E=exp(Y), where Y 
    is z-score standardized) by a logistic distribution. 
    You can adjust the distribution by passing in a different dropout
    probability distribution dropout_dist. A dropout will be modelled by random
    draws of the dropout distribution being bigger then dropout_p. 
    """
    n_data = Xsim.shape[0]
    Y = np.empty((n_data, p_dims))

    splits = np.random.choice(p_dims, replace=False, size=num_classes)
    splits.sort()

    #for sub in np.array_split(range(p_dims), splits):
    for sub in np.array_split(range(p_dims), splits):
        ky_sim = (GPy.kern.RBF(1, variance=1e-8, lengthscale=5, active_dims=[2]) # switch time info off (variance of 1e-8)
                  + GPy.kern.Linear(2, variances=np.random.uniform(3, 5)) # Linear contribution
                  + GPy.kern.Matern32(2, ARD=True, variance=np.random.uniform(4, 6), lengthscale=np.random.uniform(20,25, size=2)) # long-term
                  + GPy.kern.Matern32(2, ARD=True, variance=np.random.uniform(40, 60), lengthscale=np.random.uniform(6,7, size=2)) # mid term
                  + GPy.kern.Matern32(2, ARD=True, variance=np.random.uniform(1, 2), lengthscale=np.random.uniform(1,2, size=2)) # short term
                  #+ GPy.kern.LogisticBasisFuncKernel(2, np.random.uniform(0,10), variance=1, slope=1, active_dims=[1,2])
                  + GPy.kern.White(3,variance=noise_var)
                  )
        Ky_sim = ky_sim.K(np.c_[Xsim, t])
        Y[:, sub] = np.random.multivariate_normal(np.zeros(n_data), Ky_sim, 5).T.dot(np.random.normal(0,1,(5, sub.size)))
    #Ky_sim = ky_sim.K(np.c_[Xsim, t])
    #Y = np.random.multivariate_normal(np.zeros(n_data), Ky_sim, p_dims).T
    Y -= Y.mean(0)
    Y /= Y.std(0)
        
    # put dropouts in
    
    # First exponentiate the data
    Y = np.exp(Y)
    Y -= Y.min()
    
    #we take it from an exponentially distributed stochastic variable:
    drop_fil = (dropout_dist.pdf(Y)*np.random.uniform(0,1,Y.shape)) > dropout_p
    Y[drop_fil] = 0

    # go back to normal space and normalize again:
    Y = np.log1p(Y)
    #Y -= Y.mean(0)
    #Y /= Y.std(0)
    
    return Y, drop_fil

def simulate_new_Y_qPCR(Xsim, t, p_dims, num_classes=10, noise_var=.2):
    n_data = Xsim.shape[0]
    Y = np.empty((n_data, p_dims))

    splits = np.random.choice(p_dims, replace=False, size=num_classes)
    splits.sort()

    #for sub in np.array_split(range(p_dims), splits):
    for sub in np.array_split(range(p_dims), splits):
        ky_sim = (GPy.kern.RBF(1, variance=1e-6, lengthscale=5, active_dims=[2])
                  + GPy.kern.RBF(2, variance=.4, lengthscale=np.random.uniform(.5,.9))
                  + GPy.kern.Linear(2, variances=np.random.uniform(.08, .15))
                  + GPy.kern.RBF(2, ARD=True, variance=1, lengthscale=np.random.uniform(4,10, size=2))
                  #+ GPy.kern.LogisticBasisFuncKernel(2, np.random.uniform(0,10), variance=1, slope=1, active_dims=[1,2])
                  + GPy.kern.White(3,variance=noise_var)
                  )
        Ky_sim = ky_sim.K(np.c_[Xsim, t])
        Y[:, sub] = np.random.multivariate_normal(np.zeros(n_data), Ky_sim, 2).T.dot(np.random.normal(0,1,(2, sub.size)))
    #Ky_sim = ky_sim.K(np.c_[Xsim, t])
    #Y = np.random.multivariate_normal(np.zeros(n_data), Ky_sim, p_dims).T
    # normalize
    Y -= Y.mean(0)
    Y /= Y.std(0)
    
    return Y

def qpcr_simulation(p_genes=48, n_divisions=6, num_classes=48, seed=None, split_prob=.01):
    t, labels, seed = make_cell_division_times(n_divisions, n_replicates=9, seed=seed, std=.01, drop_p=.6)
    c = np.log2(labels) / n_divisions
    #c = t
    xvar = .7
    Xsim, seed, labels, t = simulate_latent_space(t, labels, var=xvar, seed=seed, split_prob=split_prob, gap=1.5)
    def simulate_new():
        return simulate_new_Y_qPCR(Xsim, t, p_genes, num_classes=num_classes, noise_var=.2)
    return Xsim, simulate_new, t, c, labels, seed

guo_simulation = qpcr_simulation


def rnaseq_simulation(p_genes=220, n_divisions=6, num_classes=100, seed=None, split_prob=.01, dropout_dist=stats.logistic(0, 1), dropout_p=.1):
    t, labels, seed = make_cell_division_times(n_divisions, n_replicates=9, seed=seed, std=.05, drop_p=.6)
    c = np.log2(labels) / n_divisions
    #c = t
    xvar = 1.
    Xsim, seed, labels, t = simulate_latent_space(t, labels, var=xvar, seed=seed, split_prob=split_prob, gap=1.)
    def simulate_new():
        return simulate_new_Y_RNAseq(Xsim, t, p_genes, num_classes=num_classes, noise_var=.2, dropout_dist=dropout_dist, dropout_p=dropout_p)
    return Xsim, simulate_new, t, c, labels, seed

# def guo_simulation_old(p_dims=48, n_divisions=6, seed=None):
#     t, labels, seed = make_cell_division_times(n_divisions, n_replicates=9, seed=seed, std=.03, drop_p=.6)
#     c = np.log2(labels) / n_divisions
#     #c = t
#     xvar = .6
#     Xsim, seed, labels = simulate_latent_space(t, labels, var=xvar, seed=seed, split_prob=.01)
#     def simulate_new():
#         return simulate_new_Y(Xsim, t, p_dims, num_classes=48, noise_var=.7)
#     return Xsim, simulate_new, t, c, labels, seed
