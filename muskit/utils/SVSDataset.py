from xxx


def _get_spectrograms():
    """Parse the wave file in `fpath` and.

    Returns normalized melspectrogram and linear spectrogram.
    Args:
      fpath: A string. The full path of a sound file.
    Returns:
      mel: A 2d array of shape (T, n_mels) and dtype of float32.
      mag: A 2d array of shape (T, 1+n_fft/2) and dtype of float32.
    """


def _load_sing_quality(quality_file, standard=3):
    """_load_sing_quality."""


def _phone2char(phones, char_max_len):
    """_phone2char."""


def _Hz2Semitone(freq):
    """_Hz2Semitone."""


def _full_semitone_list(semitone_min, semitone_max):
    """_full_semitone_list."""


def _calculate_phone_element_freq(phone_array):
    """Return the phone list and freq of given phone_array."""


def _phone_shift(phone_array, phone_shift_size):



def _pitch_shift(f0_array, semitone_list):



class SVSCollator(object):
    """SVSCollator."""

    def __init__(
        self,
    ):
        """init."""

    def __call__(self,):
        """call."""



class SVSDataset(Dataset):
    """SVSDataset."""

    def __init__(
        self,
    ):
        """init."""
 
    def __len__(self):
        """len."""

    def __getitem__(self,):
        """getitem."""
