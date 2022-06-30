from .loaders import *
#from iterators import *
from ..core import get_logger
from .midi_instrument_map import *
from .speech import EnglishSpeechCorpus
logger = get_logger()
from .music_loaders import *
try:
    from .music_loaders import *
except ImportError:
    logger.info("WARNING: Unable to import music related support libraries, skipping music loaders...")
