import numpy as np

class ListIterator(object):
    def __init__(self, list_of_iteration_args, batch_size,
                 random_state=None, infinite_iterator=False):
        """
        one_hot_size
        should be either None, or a list of one hot size desired
        same length as list_of_iteration_args
        list_of_iteration_args = [my_image_data, my_label_data]
        one_hot_size = [None, 10]
        """
        self.list_of_iteration_args = list_of_iteration_args
        self.batch_size = batch_size

        self.infinite_iterator = infinite_iterator
        replace = True if self.infinite_iterator else False

        self.random_state = random_state
        if random_state is None:
            raise ValueError("Must pass random state for random selection")

        iteration_args_lengths = []
        iteration_args_dims = []
        for n, ts in enumerate(list_of_iteration_args):
            c = [(li, np.array(tis).shape) for li, tis in enumerate(ts)]
            if len(iteration_args_lengths) > 0:
                if len(c[-1][1]) == 0:
                    raise ValueError("iteration_args arguments should be at least 2D arrays, detected 1D")
                # +1 to handle len vs idx offset
                if c[-1][0] + 1 != iteration_args_lengths[-1]:
                    raise ValueError("iteration_args arguments should have the same iteration length (dimension 0)")
                #if c[-1][1] != iteration_args_dims[-1]:
                #    from IPython import embed; embed(); raise ValueError()

            iteration_args_lengths.append(c[-1][0] + 1)
            iteration_args_dims.append(c[-1][1])
        self.iteration_args_lengths_ = iteration_args_lengths
        self.iteration_args_dims_ = iteration_args_dims

        # set up the indices selected for the first batch
        self.indices_ = self.random_state.choice(self.iteration_args_lengths_[0],
                                                 size=(batch_size,), replace=False)

    def next(self):
        next_batches = []
        for l in range(len(self.list_of_iteration_args)):
            t = np.zeros([self.batch_size] + list(self.iteration_args_dims_[l]), dtype=np.float32)
            for bi in range(self.batch_size):
                t[bi] = self.list_of_iteration_args[l][self.indices_[bi]]
            next_batches.append(t)
        self.indices_ = self.random_state.choice(self.iteration_args_lengths_[0],
                                                 size=(self.batch_size,), replace=False)
        return next_batches

    def __next__(self):
        return self.next()

    def __iter__(self):
        while True:
            yield next(self)
