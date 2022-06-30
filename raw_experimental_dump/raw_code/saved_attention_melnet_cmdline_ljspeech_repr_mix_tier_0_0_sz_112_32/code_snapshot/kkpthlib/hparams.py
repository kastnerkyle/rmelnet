import collections

class HParams(object):
    """
    skeletonized HParams container
    """
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __getitem__(self, key):
        return getattr(self, key)

    def __repr__(self):
        #[:-1] to drop trailing ","
        return "HParams(\n" + "\n".join([str(k).replace("'", "") + "=" + "{}".format(v) + "," for k, v in self.__dict__.items()])[:-1] + "\n)"
