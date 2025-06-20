import re
import os
import pandas as pd

# Matches lines like "12/31/2020, 23:59 -"
DATE_PATTERN = re.compile(r'^(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{2}(?: [APMapm]{2})?) -')

def load_chat(source) -> pd.DataFrame:
    """
    Accepts:
      - a file path (str or os.PathLike)
      - a file-like object (UploadedFile, StringIO, etc.)
    Returns a DataFrame with columns ['date','time','contact','message'].
    """
    close_fp = False
    if isinstance(source, (str, os.PathLike)):
        fp = open(source, encoding='utf-8')
        close_fp = True
    else:
        fp = source  # in-memory file-like

    # Skip the WhatsApp header line
    fp.readline()

    data, buf = [], []
    date = time = author = None

    for raw in fp:
        line = raw.decode('utf-8', errors='ignore') if isinstance(raw, bytes) else raw
        line = line.strip()
        if DATE_PATTERN.match(line):
            if buf:
                data.append([date, time, author, ' '.join(buf)])
            buf.clear()
            dt, msg = line.split(' - ', 1)
            date_str, time_str = dt.split(', ')
            date, time = date_str.strip(), time_str.strip()
            if ': ' in msg:
                author, message = msg.split(': ', 1)
            else:
                author, message = None, msg
            buf.append(message)
        else:
            buf.append(line)

    # flush last message
    if buf:
        data.append([date, time, author, ' '.join(buf)])

    df = pd.DataFrame(data, columns=['date','time','contact','message'])
    # parse with explicit format to avoid warnings
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y', dayfirst=True, errors='coerce')

    if close_fp:
        fp.close()
    return df
