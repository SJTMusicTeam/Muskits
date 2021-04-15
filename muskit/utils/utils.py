import xxx


def collect_stats(train_loader, args):
    """collect_stats."""


def train_one_epoch():
    """train_one_epoch."""


def validate_one_epoch():
    """validate_one_epoch."""

 
def train_one_epoch_discriminator():
    """train_one_epoch_discriminator."""


def validate_one_epoch_discriminator():
    """validate_one_epoch_discriminator."""


def save_checkpoint():
    """save_checkpoint."""


def save_model():
    """save_model."""


def record_info():
    """record_info."""


def spectrogram2wav():
    """Generate wave file from linear magnitude spectrogram.

    Args:
      mag: A numpy array of (T, 1+n_fft//2)
    Returns:
      wav: A 1-D numpy array.
    """


def log_figure():
    """log_figure."""
    # only get one sample from a batch
    # save wav and plot spectrogram


def load_wav(path):
    """Load wav."""


def save_wav(x, path):
    """Save wav."""


def normalize(S):
    """Normalize."""


if __name__ == "__main__":
