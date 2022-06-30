from .core import get_logger
import numpy as np
import copy

class ListIterator(object):
    # iterate group of sequences
    # no_random means to directly use the order of the input sequences
    # single_shuffle generates random minibatch ordering one time, repeats it
    # infinite_iterator means it will never raise StopIteration
    # random_state
    def __init__(self, list_of_sequences, batch_size,
                 no_random=False,
                 single_shuffle=False,
                 infinite_iterator=False,
                 truncate_if_ragged_last_batch=True,
                 random_state=None):
        if random_state is None:
            if no_random is False: raise ValueError("Must pass random state to use random iteration! Otherwise, set no_random=True")
            self.random_state = random_state
        self.list_of_sequences = list_of_sequences
        self.logger = get_logger()

        base_len = len(list_of_sequences[0])
        if len(list_of_sequences) > 0:
            for i in range(len(list_of_sequences)):
                if len(list_of_sequences[i]) != base_len:
                    raise ValueError("Sequence lengths for iteration do not match! Check element {} and {} of list_of_sequences".format(0, i))

        self.base_len = base_len
        self.is_ragged = False
        if (self.base_len % batch_size) != 0:
           self.is_ragged = True
           if not truncate_if_ragged_last_batch:
                self.logger.info("WARNING: batch_size for ListIterator is not evenly divisible, providing uneven last batch due to truncate_if_ragged_last_batch=False")

        self.no_random = no_random
        self.batch_size = batch_size
        self.single_shuffle = single_shuffle
        self.infinite_iterator = infinite_iterator
        self.truncate_if_ragged_last_batch = truncate_if_ragged_last_batch
        self.random_state = random_state

        batch_indices = np.arange(base_len)
        if self.no_random is False:
            self.random_state.shuffle(batch_indices)
        batches = [batch_indices[i:i + batch_size] for i in range(0, len(batch_indices), batch_size)]
        if self.is_ragged:
            if self.truncate_if_ragged_last_batch:
                batches = batches[:-1]

        self.current_batches_ = batches
        self.batches_index_ = 0

    def next(self):
        return self.__next__()

    def __iter__(self):
        while True:
            yield next(self)

    def __next__(self):
        if self.batches_index_ >= len(self.current_batches_):
            # reset and raise StopIteration
            self.batches_index_ = 0
            if self.single_shuffle is False:
                batch_indices = np.arange(self.base_len)
                if self.no_random is False:
                    self.random_state.shuffle(batch_indices)
                batches = [batch_indices[i:i + self.batch_size] for i in range(0, len(batch_indices), self.batch_size)]
                if self.is_ragged:
                    if self.truncate_if_ragged_last_batch:
                        batches = batches[:-1]
                self.current_batches_ = batches
            if self.infinite_iterator:
                pass
            else:
                raise StopIteration("End of sequence")
        i = self.current_batches_[self.batches_index_]
        this_batch = [np.array([ls[_ii] for _ii in i]) for ls in self.list_of_sequences]
        self.batches_index_ += 1
        return this_batch


class StepIterator(object):
    # iterate group of sequences
    # no_random means to directly use the order of the input sequences
    # single_shuffle generates random minibatch ordering one time, repeats it
    # infinite_iterator means it will never raise StopIteration
    # random_state
    def __init__(self, list_of_sequences, slice_size=1,
                 step_size=1,
                 random_shuffle=False,
                 circular_rotation=False,
                 reorder_once=False,
                 infinite_iterator=False,
                 truncate_if_ragged_last_batch=True,
                 random_state=None):
        if random_state is None:
            raise ValueError("Must pass random state to StepIterator!")
        self.random_state = random_state
        self.list_of_sequences = list_of_sequences
        self.logger = get_logger()

        base_len = len(list_of_sequences[0])
        if len(list_of_sequences) > 0:
            for i in range(len(list_of_sequences)):
                if len(list_of_sequences[i]) != base_len:
                    raise ValueError("Sequence lengths for iteration do not match! Check element {} and {} of list_of_sequences".format(0, i))

        self.base_len = base_len
        self.is_ragged = False
        if (self.base_len % step_size) != 0:
           self.is_ragged = True
           if not truncate_if_ragged_last_batch:
                self.logger.info("WARNING: step_size for OrderedIterator is not evenly divisible, providing uneven last batch due to truncate_if_ragged_last_batch=False")

        self.slice_size = slice_size
        self.step_size = step_size
        self.random_shuffle = random_shuffle
        self.circular_rotation = circular_rotation
        self.reorder_once = reorder_once
        self.infinite_iterator = infinite_iterator
        self.truncate_if_ragged_last_batch = truncate_if_ragged_last_batch

        batch_indices = np.arange(base_len)
        if self.random_shuffle is True:
            self.random_state.shuffle(batch_indices)
        if self.circular_rotation is True:
            rotate_point = random_state.randint(base_len)

        if self.is_ragged:
            if self.truncate_if_ragged_last_batch:
                batch_indices = batch_indices[:len(batch_indices) - len(batch_indices) % min(1, self.slice_size - self.step_size) + self.step_size]
        self.index_ = 0
        self.batch_indices = batch_indices

    def next(self):
        return self.__next__()

    def __iter__(self):
        while True:
            yield next(self)

    def __next__(self):
        if self.index_ >= self.base_len:
            # reset and raise StopIteration
            self.index_ = 0
            if self.reorder_once is False:
                batch_indices = np.arange(self.base_len)
                if self.random_shuffle is True:
                    self.random_state.shuffle(batch_indices)

                if self.circular_rotation is True:
                    rotate_point = self.random_state.randint(self.base_len)

                if self.is_ragged:
                    if self.truncate_if_ragged_last_batch:
                        batch_indices = batch_indices[:len(batch_indices) - len(batch_indices) % min(1, self.slice_size - self.step_size) + self.step_size]
                self.batch_indices = batch_indices
            raise StopIteration("End of sequence")
        else:
            i = self.index_
            this_batch = [ls[i:i + self.slice_size] for ls in self.list_of_sequences]
            if self.slice_size == 1:
                this_batch = [t[0] for t in this_batch]
            self.index_ += self.step_size
            if len(self.list_of_sequences) == 1:
                 return this_batch[0]
        return this_batch


class CyclicListIterator(object):
    # iterate list of data 
    # assume we want to cyclic wrap, and reset randomly to a new start point after each full loop
    # preserve order of the data (for things like transformer xl / rnn)
    # infinite_iterator means it will never raise StopIteration
    def __init__(self, list_of_data_lists,
                 batch_size,
                 sequence_length,
                 overlap,
                 eof_split_value=None,
                 circular_rotation=True,
                 infinite_iterator=True,
                 truncate_if_ragged_last_batch=True,
                 random_state=None):
        if random_state is None:
            raise ValueError("Must pass random state to CyclicListIterator!")
        self.random_state = random_state
        self.list_of_data_lists = list_of_data_lists
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.overlap = overlap
        self.eof_split_value = eof_split_value

        self.circular_rotation = circular_rotation

        self.logger = get_logger()

        if self.circular_rotation != True:
            raise ValueError("Current logic only supports circular rotation!")

        base_len = len(list_of_data_lists[0])
        if len(list_of_data_lists) > 0:
            for i in range(len(list_of_data_lists)):
                if len(list_of_data_lists[i]) != base_len:
                    raise ValueError("Sequence lengths for iteration do not match! Check element {} and {} of list_of_flat_data_lists".format(0, i))

        self.base_len = base_len
        self.is_ragged = False
        if (self.base_len % self.overlap) != 0:
           self.is_ragged = True
           if not truncate_if_ragged_last_batch:
                self.logger.info("WARNING: step_size for CyclicListIterator is not evenly divisible, providing uneven last batch due to truncate_if_ragged_last_batch=False")

        self.infinite_iterator = infinite_iterator
        self.truncate_if_ragged_last_batch = truncate_if_ragged_last_batch

        self._all_start_points = [n for n, el in enumerate(list_of_data_lists[0]) if el == self.eof_split_value]
        self._this_list_of_data_lists = copy.deepcopy(self.list_of_data_lists)
        # double the data, so no matter what start point is used we can traverse the whole data 1 time
        self._this_list_of_data_lists = [copy.deepcopy(l) + copy.deepcopy(l) for l in self._this_list_of_data_lists]
        if self.base_len < self.batch_size:
            raise ValueError("Data length of less than batch_size detected!")
        if self.base_len < self.sequence_length:
            raise ValueError("Data length of less than sequence_length detected!")
        # trick: fold the iterated indices to get start points, will make them evenly split
        if len(self._all_start_points) < 1:
            # if there is no eof_split points, just start on random start points from anywhere in the data
            self._this_start_points = copy.deepcopy(list(range(0, self.base_len)))
        else:
            if len(self._all_start_points) < self.batch_size:
                raise ValueError("Less values matching self.eof_split_value than batch_size detected!")
            # if we have eof split points, only allow start on the eof points
            all_possible_start_points = np.array(self._all_start_points)
            if 0 not in self._all_start_points:
                all_possible_start_points = np.concatenate(([0,], all_possible_start_points))
            self._this_start_points = copy.deepcopy(self._all_start_points)

        # this is how we reset the iterator
        self._epoch_start_points = copy.deepcopy(self._this_start_points)
        random_state.shuffle(self._epoch_start_points)
        self._epoch_start_points = self._epoch_start_points[:self.batch_size]
        assert len(self._epoch_start_points) == self.batch_size
        # shuffle the start points so there are no batches which always start at the same place...

        # trace the number of steps taken, once this breaches self.base_len, will reset the whole index schema
        self.step_ = 0

    def next(self):
        return self.__next__()

    def __iter__(self):
        while True:
            yield next(self)

    def __next__(self):
        if self.step_ >= self.base_len:
            # reset and raise StopIteration
            self.step_ = 0
            self._epoch_start_points = copy.deepcopy(self._this_start_points)
            self.random_state.shuffle(self._epoch_start_points)
            self._epoch_start_points = self._epoch_start_points[:self.batch_size]
            assert len(self._epoch_start_points) == self.batch_size
            raise StopIteration("End of sequence")
        else:
            i = self.step_
            this_multi_batch = []
            for j in range(len(self._this_list_of_data_lists)):
                this_batch = []
                for b in range(self.batch_size):
                    ls = self._this_list_of_data_lists[j]
                    ss = self._epoch_start_points[b] + i
                    non_eof_values = [el for _n, el in enumerate(ls[ss:ss + 2 * self.sequence_length]) if el != self.eof_split_value]
                    tb = np.array(non_eof_values)[:self.sequence_length]
                    assert len(tb) == self.sequence_length
                    this_batch.append(tb)
                this_batch = np.array(this_batch).T
                this_multi_batch.append(this_batch)
            if len(this_multi_batch) == 1:
                out_batch = this_multi_batch[0][..., None]
            else:
                np.concatenate([tmb[..., None] for tmb in this_multi_batch], axis=-1)
            self.step_ += self.sequence_length - self.overlap
        return out_batch
