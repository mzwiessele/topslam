'''
Created on 1 Jun 2016

@author: maxz
'''
import unittest, numpy as np, os, GPy
from matplotlib import cbook, pyplot as plt
from nose import SkipTest

extensions = ['npz']
basedir = os.path.dirname(os.path.relpath(os.path.abspath(__file__)))


def _image_directories():
    """
    Compute the baseline and result image directories for testing *func*.
    Create the result directory if it doesn't exist.
    """
    #module_name = __init__.__module__
    #mods = module_name.split('.')
    #basedir = os.path.join(*mods)
    result_dir = os.path.join(basedir, 'testresult','.')
    baseline_dir = os.path.join(basedir, 'baseline','.')
    if not os.path.exists(result_dir):
        cbook.mkdirs(result_dir)
    return baseline_dir, result_dir

baseline_dir, result_dir = _image_directories()
if not os.path.exists(baseline_dir):
    raise SkipTest("Not installed from source, baseline not available. Install from source to test plotting")

def _image_comparison(baseline_images, extensions=['pdf','svg','png'], tol=11, rtol=1e-3, **kwargs):

    for num, base in zip(plt.get_fignums(), baseline_images):
        for ext in extensions:
            fig = plt.figure(num)
            fig.canvas.draw()
            #fig.axes[0].set_axis_off()
            #fig.set_frameon(False)
            if ext in ['npz']:
                figdict = flatten_axis(fig)
                np.savez_compressed(os.path.join(result_dir, "{}.{}".format(base, ext)), **figdict)
                fig.savefig(os.path.join(result_dir, "{}.{}".format(base, 'png')),
                            transparent=True,
                            edgecolor='none',
                            facecolor='none',
                            #bbox='tight'
                            )
    for num, base in zip(plt.get_fignums(), baseline_images):
        for ext in extensions:
            #plt.close(num)
            actual = os.path.join(result_dir, "{}.{}".format(base, ext))
            expected = os.path.join(baseline_dir, "{}.{}".format(base, ext))
            if ext == 'npz':
                def do_test():
                    if not os.path.exists(expected):
                        import shutil
                        shutil.copy2(actual, expected)
                        #shutil.copy2(os.path.join(result_dir, "{}.{}".format(base, 'png')), os.path.join(baseline_dir, "{}.{}".format(base, 'png')))
                        raise IOError("Baseline file {} not found, copying result {}".format(expected, actual))
                    else:
                        exp_dict = dict(np.load(expected).items())
                        act_dict = dict(np.load(actual).items())
                        for name in act_dict:
                            if name in exp_dict:
                                try:
                                    np.testing.assert_allclose(exp_dict[name], act_dict[name], err_msg="Mismatch in {}.{}".format(base, name), rtol=rtol, **kwargs)
                                except AssertionError as e:
                                    raise SkipTest(e)
            yield do_test
    plt.close('all')

def flatten_axis(ax, prevname=''):
    import inspect
    members = inspect.getmembers(ax)

    arrays = {}

    def _flatten(l, pre):
        arr = {}
        if isinstance(l, np.ndarray):
            if l.size:
                arr[pre] = np.asarray(l)
        elif isinstance(l, dict):
            for _n in l:
                _tmp = _flatten(l, pre+"."+_n+".")
                for _nt in _tmp.keys():
                    arrays[_nt] = _tmp[_nt]
        elif isinstance(l, list) and len(l)>0:
            for i in range(len(l)):
                _tmp = _flatten(l[i], pre+"[{}]".format(i))
                for _n in _tmp:
                    arr["{}".format(_n)] = _tmp[_n]
        else:
            return flatten_axis(l, pre+'.')
        return arr


    for name, l in members:
        if isinstance(l, np.ndarray):
            arrays[prevname+name] = np.asarray(l)
        elif isinstance(l, list) and len(l)>0:
            for i in range(len(l)):
                _tmp = _flatten(l[i], prevname+name+"[{}]".format(i))
                for _n in _tmp:
                    arrays["{}".format(_n)] = _tmp[_n]

    return arrays

def _a(x,y,decimal):
    np.testing.assert_array_almost_equal(x, y, decimal)

def compare_axis_dicts(x, y, decimal=6):
    try:
        assert(len(x)==len(y))
        for name in x:
            _a(x[name], y[name], decimal)
    except AssertionError as e:
        raise SkipTest(e.message)


extensions = ['npz']


def test_topslam():
    np.random.seed(111)
    import matplotlib
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    #matplotlib.rcParams[u'figure.figsize'] = (4,3)
    matplotlib.rcParams[u'text.usetex'] = False
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        #from topslam.simulation.simulate_trajectory import qpcr_simulation

        #Xsim, simulate_new, t, c, labels, seed = qpcr_simulation(48, 6, 5001)

        #np.random.seed(3)
        #Y = simulate_new()

        #m = GPy.models.BayesianGPLVM(Y, 2, X=Xsim, num_inducing=25)
        #m.optimize()

        try:
            test_data = np.load(os.path.join(basedir, 'test_data_model.npz'))
        except IOError:
            raise #SkipTest('not installed by source, skipping plotting tests')
        labels = test_data['labels']

        m = GPy.models.BayesianGPLVM(test_data['Y'].copy(), 2, num_inducing=25, initialize=False)
        m.param_array[:] = test_data['model_params']
        m.initialize_parameter()


        from topslam import ManifoldCorrectionKNN
        mc = ManifoldCorrectionKNN(m, 10)

        fig, ax = plt.subplots()
        mc.plot_waddington_landscape(ax=ax)
        mc.plot_graph_nodes(ax=ax)

        mc.plot_time_graph()

        mc.plot_time_graph(labels)

        fig, ax = plt.subplots()
        mc.plot_graph_nodes(labels, ax=ax)
        mc.plot_graph_labels(labels, ax=ax, adjust=False, box=False)

        fig, ax = plt.subplots()
        mc.plot_graph_nodes(ax=ax)
        mc.plot_graph_labels(labels, ax=ax, adjust=True, box=True)

    for do_test in _image_comparison(baseline_images=['topslam_{}'.format(sub) for sub in ["waddington_nodes",
                                                                                           "time_tree_no_lab",
                                                                                           "time_tree_labels",
                                                                                           'graph_nodes_labels_nobox_noadjust',
                                                                                           'graph_nodes_nolabs_box_adjust']
                                                      ], extensions=extensions):
        yield do_test
        
        
def test_other():
    np.random.seed(111)
    import matplotlib
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    #matplotlib.rcParams[u'figure.figsize'] = (4,3)
    matplotlib.rcParams[u'text.usetex'] = False
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        #from topslam.simulation.simulate_trajectory import qpcr_simulation

        #Xsim, simulate_new, t, c, labels, seed = qpcr_simulation(48, 6, 5001)

        #np.random.seed(3)
        #Y = simulate_new()

        #m = GPy.models.BayesianGPLVM(Y, 2, X=Xsim, num_inducing=25)
        #m.optimize()

        try:
            test_data = np.load(os.path.join(basedir, 'test_data_model.npz'))
            test_init = np.load(os.path.join(basedir, 'test_data_others.npz'))
        except IOError:
            raise #SkipTest('not installed by source, skipping plotting tests')
        labels = test_data['labels']
        dims = test_init['dims'].tolist()
        X_init = test_init['X_init']
        
        m = GPy.models.BayesianGPLVM(test_data['Y'].copy(), 2, num_inducing=25, initialize=False)
        m.param_array[:] = test_data['model_params']
        m.initialize_parameter()

        from topslam import ManifoldCorrectionKNN
        mc = ManifoldCorrectionKNN(m, 10)

        from topslam.plotting import plot_comparison, plot_dist_hist, plot_labels_other, plot_landscape_other
        plot_comparison(mc, X_init, dims, labels, np.unique(labels), 0)

        plot_dist_hist(test_data['Y'])
        
        X, pt = X_init[:, dims['t-SNE']], test_data['t']
        fig, ax = plt.subplots()
        plot_landscape_other(X, pt, labels, ax=ax)
        plot_labels_other(X, pt, labels, ax=ax)

    for do_test in _image_comparison(baseline_images=['other_{}'.format(sub) for sub in ["comparison",
                                                                                           "dist_hist",
                                                                                           "landscape_labs",
                                                                                           ]
                                                      ], extensions=extensions):
        yield do_test
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testPlotWaddington']
    unittest.main()