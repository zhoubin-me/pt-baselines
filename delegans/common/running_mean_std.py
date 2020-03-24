import tensorflow as tf
import numpy as np

class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count






def test_runningmeanstd():
    for (x1, x2, x3) in [
        (np.random.randn(3), np.random.randn(4), np.random.randn(5)),
        (np.random.randn(3,2), np.random.randn(4,2), np.random.randn(5,2)),
        ]:

        rms = RunningMeanStd(epsilon=0.0, shape=x1.shape[1:])

        x = np.concatenate([x1, x2, x3], axis=0)
        ms1 = [x.mean(axis=0), x.var(axis=0)]
        rms.update(x1)
        rms.update(x2)
        rms.update(x3)
        ms2 = [rms.mean, rms.var]

        np.testing.assert_allclose(ms1, ms2)

def test_tf_runningmeanstd():
    for (x1, x2, x3) in [
        (np.random.randn(3), np.random.randn(4), np.random.randn(5)),
        (np.random.randn(3,2), np.random.randn(4,2), np.random.randn(5,2)),
        ]:

        rms = TfRunningMeanStd(epsilon=0.0, shape=x1.shape[1:], scope='running_mean_std' + str(np.random.randint(0, 128)))

        x = np.concatenate([x1, x2, x3], axis=0)
        ms1 = [x.mean(axis=0), x.var(axis=0)]
        rms.update(x1)
        rms.update(x2)
        rms.update(x3)
        ms2 = [rms.mean, rms.var]

        np.testing.assert_allclose(ms1, ms2)


def profile_tf_runningmeanstd():
    import time
    from baselines.common import tf_util

    tf_util.get_session( config=tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1,
        allow_soft_placement=True
    ))

    x = np.random.random((376,))

    n_trials = 10000
    rms = RunningMeanStd()
    tfrms = TfRunningMeanStd()

    tic1 = time.time()
    for _ in range(n_trials):
        rms.update(x)

    tic2 = time.time()
    for _ in range(n_trials):
        tfrms.update(x)

    tic3 = time.time()

    print('rms update time ({} trials): {} s'.format(n_trials, tic2 - tic1))
    print('tfrms update time ({} trials): {} s'.format(n_trials, tic3 - tic2))


    tic1 = time.time()
    for _ in range(n_trials):
        z1 = rms.mean

    tic2 = time.time()
    for _ in range(n_trials):
        z2 = tfrms.mean

    assert z1 == z2

    tic3 = time.time()

    print('rms get mean time ({} trials): {} s'.format(n_trials, tic2 - tic1))
    print('tfrms get mean time ({} trials): {} s'.format(n_trials, tic3 - tic2))



    '''
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE) #pylint: disable=E1101
    run_metadata = tf.RunMetadata()
    profile_opts = dict(options=options, run_metadata=run_metadata)



    from tensorflow.python.client import timeline
    fetched_timeline = timeline.Timeline(run_metadata.step_stats) #pylint: disable=E1101
    chrome_trace = fetched_timeline.generate_chrome_trace_format()
    outfile = '/tmp/timeline.json'
    with open(outfile, 'wt') as f:
        f.write(chrome_trace)
    print('Successfully saved profile to {}. Exiting.'.format(outfile))
    exit(0)
    '''



if __name__ == '__main__':
   profile_tf_runningmeanstd()