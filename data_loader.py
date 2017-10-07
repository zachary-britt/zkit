import pandas as pd

def data_loader(filename, proportion=1.0):

    if proportion != 1.0:

        # nth row to keep
        n = int(1.0 / proportion)

        # length of dataset
        row_count = sum(1 for row in open(filename))

        # Row indices to skip
        skipped = [x for x in range(1, num_lines) if x % n != 0]
    else:
        skipped = None

    df = pd.read_csv(filename, skiprows=skipped)
    return df
