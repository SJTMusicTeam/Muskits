import xxx


def collect_stats(train_loader, args):
    """collect_stats."""
    pass


def train_one_epoch():
    """train_one_epoch."""
    pass


def validate_one_epoch():
    """validate_one_epoch."""
    pass


def train_one_epoch_discriminator():
    """train_one_epoch_discriminator."""
    pass


def validate_one_epoch_discriminator():
    """validate_one_epoch_discriminator."""
    pass


def save_checkpoint():
    """save_checkpoint."""
    pass


def save_model():
    """save_model."""
    pass


def record_info():
    """record_info."""
    pass


def spectrogram2wav():
    """Generate wave file from linear magnitude spectrogram.

    Args:
      mag: A numpy array of (T, 1+n_fft//2)
    Returns:
      wav: A 1-D numpy array.
    """
    pass


def log_figure():
    """log_figure."""
    # only get one sample from a batch
    # save wav and plot spectrogram
    pass


def load_wav(path):
    """Load wav."""
    pass


def save_wav(x, path):
    """Save wav."""
    pass


def normalize(S):
    """Normalize."""
    pass


if __name__ == "__main__":
    pass
