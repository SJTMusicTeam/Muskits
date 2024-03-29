import logging
from pathlib import Path
from typing import Dict
from typing import List
from typing import Union

from typeguard import check_argument_types

# read text
def read_2column_text(path: Union[Path, str]) -> Dict[str, str]:
    """Read a text file having 2 column as dict object.
    Examples:
        wav.scp:
            key1 /some/path/a.wav
            key2 /some/path/b.wav
        >>> read_2column_text('wav.scp')
        {'key1': '/some/path/a.wav', 'key2': '/some/path/b.wav'}
    """
    assert check_argument_types()

    data = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for linenum, line in enumerate(f, 1):
            sps = line.rstrip().split(maxsplit=1)
            if len(sps) == 1:
                k, v = sps[0], ""
            else:
                k, v = sps
            if k in data:
                raise RuntimeError(f"{k} is duplicated ({path}:{linenum})")
            data[k] = v
    return data


# read duration
def load_num_sequence_text(
    path: Union[Path, str], loader_type: str = "text_float"
) -> Dict[str, List[Union[float, int]]]:
    """Read a text file indicating sequences of number
    Examples:
        key1 1 2 3
        key2 3 4 5 6
        >>> d = load_num_sequence_text('duration')
        >>> np.testing.assert_array_equal(d["key1"], np.array([0.1, 0.2, 0.3]))
    """
    assert check_argument_types()
    if loader_type == "text_int":
        delimiter = " "
        dtype = int
    elif loader_type == "text_float":
        delimiter = " "
        dtype = float
    elif loader_type == "csv_int":
        delimiter = ","
        dtype = int
    elif loader_type == "csv_float":
        delimiter = ","
        dtype = float
    elif loader_type == "str":
        delimiter = " "
        dtype = str
    else:
        raise ValueError(f"Not supported loader_type={loader_type}")

    # path looks like:
    #   utta 1,0
    #   uttb 3,4,5
    # -> return {'utta': np.ndarray([1, 0]),
    #            'uttb': np.ndarray([3, 4, 5])}
    d = read_2column_text(path)

    # Using for-loop instead of dict-comprehension for debuggability
    retval = {}
    for k, v in d.items():
        try:
            retval[k] = [dtype(i) for i in v.split(delimiter)]
        except TypeError:
            logging.error(f'Error happened with path="{path}", id="{k}", value="{v}"')
            raise
    return retval


def load_label_sequence(path: Union[Path, str]) -> Dict[str, List[Union[float, int]]]:
    """Read a text file indicating sequences of number
    Examples:
        key1 1 2 3
        key2 3 4 5 6
        >>> d = load_num_sequence_text('duration')
        >>> np.testing.assert_array_equal(d["key1"], [0.1, 0.2, "啊"]))
    """
    assert check_argument_types()

    delimiter = " "
    dtype = float

    d = read_2column_text(path)

    retval = {}
    for k, v in d.items():
        try:
            start, end, text = v.split(delimiter)
            retval[k] = [dtype(start), dtype(end), text]
        except TypeError:
            logging.error(f'Error happened with path="{path}", id="{k}", value="{v}"')
            raise
    return retval


def read_label(path: Union[Path, str]) -> Dict[str, List[Union[float, int]]]:
    """Read a text file indicating sequences of number
    Examples:
        key1 start_time_1 end_time_1 phone_1 start_time_2 end_time_2 phone_2 ....\n
        key2 start_time_1 end_time_1 phone_1 \n
        >>> d = load_num_sequence_text('label')
        >>> np.testing.assert_array_equal(d["key1"], [0.1, 0.2, "啊"]))
    """
    assert check_argument_types()
    label = open(path, "r", encoding="utf-8")

    retval = {}
    for label_line in label.readlines():
        line = label_line.strip().split()
        key = line[0]
        phn_info = line[1:]
        temp_info = []
        for i in range(len(phn_info) // 3):
            temp_info.append(
                [phn_info[i * 3], phn_info[i * 3 + 1], phn_info[i * 3 + 2]]
            )
        retval[key] = temp_info
    return retval
