"""
**sns_rad.binary** module provides access to errant beam data stored in binary files.<br />
"""
import gzip
from array import array
from datetime import datetime as dt
import os
from io import IOBase, BytesIO
from pathlib import Path

HEADER = 'SNS-BIET2020'
HEADER_SZ = 12

ENDRCRD = 254
ENDFILE = 255

NAME = 1
UNIXTST = 2
EPICSTST = 3
TAG = 4
DESCR = 5

I32PARM = 6
I32 = 16
I32WFPARM = 26
I32WF = 36

DBLPARM = 7
DBL = 17
DBLWFPARM = 27
DBLWF = 37

U8PARM = 8
U8 = 18
U8WFPARM = 28
U8WF = 38

FLTPARM = 9
FLT = 19
FLTWFPARM = 29
FLT8WF = 39


class Writer:

    def __init__(self, path):
        self.file = open(path, 'wb')
        self.file.write(HEADER)
        # print(bytes.decode())

    def close(self):
        self.file.close()


class ReadStorage:
    MAX_COMPRESSED_TO_BUFFER = 100_000_000
    def __init__(self, path, mode='auto'):
        if isinstance(path, str) or isinstance(path, Path):
            p = Path(path)
            if mode == 'auto' and p.name.endswith('.gz'):
                mode = 'gz'

            if mode == 'gz':
                if p.stat().st_size > ReadStorage.MAX_COMPRESSED_TO_BUFFER:
                    # for large archive use  file interface for the comressed data
                    self.file = gzip.GzipFile(p, 'rb')
                else:
                    # for small archives just decompress the whole archive into memory
                    self.file = BytesIO()
                    with gzip.open(p, 'rb') as f_in:
                        self.file.write(f_in.read())
            else:
                self.file = open(p, 'rb')
        elif isinstance(path, IOBase):
            self.file = path
        # bytes = self.file.read(HEADER_SZ)
        # print(bytes.decode())
        self.rec_table = read_epilog(self.file)
        self.file.seek(HEADER_SZ)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()
        return False

    def close(self):
        self.file.close()

    def read(self):
        return read_record(self.file)

    def __iter__(self):
        self.file.seek(HEADER_SZ)
        return self

    def __next__(self):
        record = read_record(self.file)
        if 'EOF' in record:
            raise StopIteration
        return record

    def readAt(self, key):
        offset = self.rec_table[key]
        self.file.seek(offset)
        return read_record(self.file)

    def __getitem__(self, key):

        if isinstance(key, slice):
            return [self.readAt(i) for i in range(key.start, key.stop, key.step)]
        else:
            return self.readAt(key)

    def __len__(self):
        return len(self.rec_table)


"""
    Internal implementation
"""


def read_size(file, long=True):
    if long:
        bytes = array('I', file.read(4))
        bytes.byteswap()
        return bytes[0]
    else:
        return read_byte(file)


def read_byte(file):
    return int.from_bytes(file.read(1), 'big')


def short_str(file):
    length = read_size(file, False)
    return file.read(length).decode()


def name(file, rec):
    length = read_size(file, False)
    raw = file.read(length)
    # print(raw)
    name = raw.decode()
    rec['name'] = name
    return True


def tst(file, rec, trim=False):
    raw = file.read(8)
    ar = array('I', raw)
    ar.byteswap()
    t = dt.fromtimestamp(ar[0] + ar[1] * 1E-9)
    rec['timestamp'] = trim_tst(t) if trim else t

    return True


def trim_tst(t):
    seconds = int(t.timestamp())
    nanos = int(((t.timestamp() % 1) * 1_000_000_000)) % 1_000_000_000
    nanos = nanos & 0xFFFF0000
    return dt.fromtimestamp(seconds + nanos * 1E-9)


def param_value(param=False, wf=False, vtype='I'):
    type_size = {
        'I': 4,
        'B': 1,
        'd': 8,
        'f': 4
    }

    def read(file, rec):
        if param:
            key = short_str(file)
            # print('key', key)
        else:
            key = 'value'
        if wf:
            length = read_size(file)
            raw = file.read(length)
            ar = array(vtype, raw)
            ar.byteswap()
            value = ar
        else:
            raw = file.read(type_size[vtype])
            ar = array(vtype, raw)
            ar.byteswap()
            value = ar[0]
        if param:
            pdict = rec.get('parameters', {})
            pdict[key] = value
            rec['parameters'] = pdict
        else:
            rec[key] = value
        return True

    return read


def description(file, rec):
    length = read_size(file, True)
    desc = file.read(length).decode()
    rec['description'] = desc
    return True


def tag(file, rec):
    length = read_size(file, False)
    tag = file.read(length).decode()
    tags = rec.get('tags', [])
    tags.append(tag)
    rec['tags'] = tags
    return True


def end_record(file, rec):
    return False


def end_file(file, rec):
    rec['EOF'] = True
    return False


parser = {
    DESCR: description,
    TAG: tag,
    NAME: name,
    UNIXTST: tst,

    I32PARM: param_value(param=True, wf=False, vtype='I'),
    I32: param_value(param=False, wf=False, vtype='I'),
    I32WF: param_value(param=False, wf=True, vtype='I'),
    I32WFPARM: param_value(param=True, wf=True, vtype='I'),

    DBLPARM: param_value(param=True, wf=False, vtype='d'),
    DBL: param_value(param=False, wf=False, vtype='d'),
    DBLWF: param_value(param=False, wf=True, vtype='d'),
    DBLWFPARM: param_value(param=True, wf=True, vtype='d'),

    U8PARM: param_value(param=True, wf=False, vtype='B'),
    U8: param_value(param=False, wf=False, vtype='B'),
    U8WF: param_value(param=False, wf=True, vtype='B'),
    U8WFPARM: param_value(param=True, wf=True, vtype='B'),

    FLTPARM: param_value(param=True, wf=False, vtype='f'),
    FLT: param_value(param=False, wf=False, vtype='f'),
    FLT8WF: param_value(param=False, wf=True, vtype='f'),
    FLTWFPARM: param_value(param=True, wf=True, vtype='f'),

    ENDRCRD: end_record,
    ENDFILE: end_file
}


def read_record(file):
    rec_size = read_size(file)
    # print(rec_size)
    # buffer = file.read(rec_size)
    record = {}

    while True:
        field_type = read_byte(file)
        # print('Field type', field_type, hex(field_type))
        func = parser.get(field_type, lambda file, rec: True)
        result = func(file, record)
        if not result:
            break
    return record


def read_epilog(file):
    OFST_SIZE = 8
    file.seek(-4, os.SEEK_END)
    raw = array('I', file.read(4))
    raw.byteswap()
    num_records = raw[0]

    file.seek(-4 - OFST_SIZE * num_records, os.SEEK_END)
    # the last record is EOF, and the very last offset has nothing
    raw = array('Q', file.read(OFST_SIZE * (num_records - 2)))
    raw.byteswap()
    # print('Number of records: ', len(raw))
    # print(raw.itemsize)
    # print(raw)
    return raw





