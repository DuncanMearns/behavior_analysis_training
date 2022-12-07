import numpy as np
import pandas as pd
from sklearn.decomposition import PCA as sklearn_pca
from pathlib import Path
from typing import Union


class PCA:

    def __init__(self, data):
        self.data = data

    @property
    def mean(self):
        """Returns the mean of the data."""
        return self.data.mean(axis=0)

    @property
    def std(self):
        """Returns the standard deviation of the data."""
        return self.data.std(axis=0)

    def whiten(self, mean=None, std=None):
        if (mean is None) and (std is None):
            whitened = (self.data - self.mean) / self.std
        elif (mean is not None) and (std is not None):
            whitened = (self.data - mean) / std
        else:
            raise ValueError('both mean and std must be specified!')
        return PCA(whitened)

    def transform(self, whiten=True, **kwargs):
        if whiten:
            data_to_transform = self.whiten(**kwargs).data
        else:
            data_to_transform = self.data
        pca = sklearn_pca()
        transformed = pca.fit_transform(data_to_transform)
        return PCA(transformed), pca

    def map(self, vectors, whiten=True, **kwargs):
        if whiten:
            data_to_map = self.whiten(**kwargs).data
        else:
            data_to_map = self.data
        assert vectors.shape[1] == data_to_map.shape[1], 'pca vector shape does not match data shape'
        mapped = np.dot(data_to_map, vectors.T)
        return PCA(mapped)


class BoutData(PCA):
    """Data object for handling bout kinematics.

    Attributes
    ----------
    data : pd.DataFrame
        Multi-indexed DataFrame containing bout data.
    metadata : pd.DataFrame
        DataFrame containing basic information about all bouts.
    """

    def __init__(self, data: pd.DataFrame, metadata: pd.DataFrame = None):
        super().__init__(data)
        assert isinstance(self.data.index, pd.MultiIndex), 'Data must be MultiIndexed!'
        self.metadata = metadata

    @classmethod
    def from_metadata(cls,
                      metadata: Union[pd.DataFrame, str, Path],
                      directory: Union[str, Path],
                      tail_only: bool = True):
        """Constructor method for generating BoutData from metadata.

        Parameters
        ----------
        metadata: pd.DataFrame | str | Path
            Either a DataFrame object containing bout metadata or a path to a corresponding csv file.
        directory: str | Path
            Top-level directory containing kinematic data.
        tail_only : bool (default = True)
            Whether or not to only keep information about tail kinematics when importing bouts.
        """
        # Open the metadata DataFrame
        if isinstance(metadata, pd.DataFrame):
            bouts_df = metadata
        elif isinstance(metadata, (str, Path)):
            bouts_df = pd.read_csv(metadata, dtype=dict(ID=str, code=str))
        else:
            raise TypeError('metadata must be path or DataFrame')
        # Get paths
        directory = Path(directory)
        paths = []
        for ID, fish_bouts in bouts_df.groupby('ID'):
            for code in pd.unique(fish_bouts['video_code']):
                paths.append(directory.joinpath(ID, code + '.csv'))
        # Import bouts
        data = import_csvs(*paths)
        # Keep desired columns
        print('Reformatting multi-indexed DataFrame...', end=' ')
        if 'tracked' in data.columns:
            data = data[data['tracked']]
        if tail_only:
            tail_columns = [col for col in data.columns if col[0] == 'k']
            data = data[tail_columns]
        # Assign bout index
        index_df = data.index.to_frame()
        video_data_dfs = []
        for code, video_bouts in bouts_df.groupby('video_code'):
            video_data = index_df.loc[(slice(None), code, slice(None)), :].copy()
            bout_index = np.empty(len(video_data)) + np.nan
            for idx, info in video_bouts.iterrows():
                bout_index[info.start:info.end + 1] = idx
            video_data['bout_index'] = bout_index
            video_data_dfs.append(video_data)
        concat = pd.concat(video_data_dfs)
        reindexed = pd.MultiIndex.from_frame(concat)
        data.index = reindexed.reorder_levels(('ID', 'video_code', 'bout_index', 'frame'))
        data = data[~data.index.get_level_values('bout_index').isna()]
        data.index.set_levels(data.index.levels[2].astype('int64'), level='bout_index', inplace=True)
        print('done!\n')
        # Return object
        return cls(data, bouts_df)

    def __str__(self):
        return self.data.__str__()

    def filter_tail_lengths(self, percentile=99):
        print('Filtering tail lengths...', end=' ')
        if 'length' not in self.data.columns:
            raise ValueError('Data des not contain "length" column.')
        long_tail = self.data[self.data['length'] > np.percentile(self.data['length'].values, percentile)]
        long_tail_bouts = long_tail.index.get_level_values('bout_index')
        metadata = self.metadata.loc[self.metadata.index[np.isin(self.metadata.index, long_tail_bouts, invert=True)]]
        data = self.data[np.isin(self.data.index.get_level_values('bout_index'), long_tail_bouts, invert=True)]
        print(f'{len(long_tail_bouts.unique())} bouts removed.\n')
        return BoutData(data, metadata)

    def tail_only(self):
        tail_columns = [col for col in self.data.columns if col[0] == 'k']
        data = self.data[tail_columns]
        return BoutData(data, self.metadata)

    def _get_from_frame(self, df):
        indexer = self.data.index.get_locs([df['ID'].values.unique(),
                                            df['video_code'].values.unique(),
                                            df.index])
        sliced = self.data.iloc[indexer, :]
        return BoutData(sliced, df)

    def _get_from_dict(self, idxs=(), **kwargs):
        index_values = dict(IDs=slice(None), codes=slice(None), bout_indices=slice(None))
        for key, val in kwargs.items():
            if len(val):
                index_values[key] = val
        indexer = self.data.index.get_locs([index_values['IDs'], index_values['video_codes'], index_values['bout_indices']])
        if len(idxs):
            indices = self.data.index.take(indexer).remove_unused_levels()
            take_bouts = indices.levels[2].take(idxs)
            take_indices = indices.get_locs([slice(None), slice(None), take_bouts])
            indexer = indexer[take_indices]
        sliced = self.data.iloc[indexer, :]
        sliced.index = sliced.index.remove_unused_levels()
        if self.metadata is not None:
            metadata = self.metadata.loc[sliced.index.levels[2]]
        else:
            metadata = None
        return BoutData(sliced, metadata)

    def get(self, IDs=(), video_codes=(), bout_indices=(), idxs=(), df=None):
        if df is not None:
            new = self._get_from_frame(df)
        else:
            new = self._get_from_dict(IDs=IDs, video_codes=video_codes, bout_indices=bout_indices, idxs=idxs)
        return new

    def iter(self, values=False, ndims=None, **kwargs):
        data = self.data
        if len(kwargs):
            data = self.get(**kwargs).data
        for idx, bout in data.groupby('bout_index'):
            if values:
                bout = bout.values
                if ndims:
                    bout = bout[:, :ndims]
                yield idx, bout
            else:
                yield idx, bout.reset_index(level=['ID', 'video_code', 'bout_index'], drop=True)

    def to_list(self, values=False, ndims=None, **kwargs):
        return [bout for i, bout in self.iter(values, ndims, **kwargs)]

    def whiten(self, **kwargs):
        whitened = super().whiten(**kwargs).data
        return BoutData(whitened, metadata=self.metadata)

    def transform(self, **kwargs):
        transformed, pca = super().transform(**kwargs)
        transformed = pd.DataFrame(transformed.data, index=self.data.index)
        return BoutData(transformed, metadata=self.metadata), pca

    def map(self, vectors, **kwargs):
        mapped = super().map(vectors, **kwargs).data
        mapped = pd.DataFrame(mapped, index=self.data.index, columns=[f'c{i}' for i in range(mapped.shape[1])])
        return BoutData(mapped, metadata=self.metadata)


def import_csvs(*args, index_levels=(1, 0), level_names=("ID", "video_code", "frame")):
    """Imports and concatenates csv files in a multi-indexed DataFrame.

    Parameters
    ----------
    args : iterable of Path or string objects
        Files or directories containing csv files to be concatenated.
    index_levels : iterable
        Iterable of ints. Each index level represents the part of the file path to be included in the multi-index. E.g.
        index level 0 indicates the filename (excluding the extension), index level 1 indicates the parent folder etc..
    level_names : iterable
        Iterable of strings. Names for each index level + the name of the index within each csv.

    Returns
    -------
    pd.DataFrame
        A multi-indexed DataFrame. Each level of the index represents a part of the file path (given by the index level)
        of the csv files included in the DataFrame."""
    try:  # check index levels and level names match
        assert len(level_names) == (len(index_levels) + 1)
    except AssertionError:
        raise ValueError("Number of level names must be one greater than number of index levels.")
    # Obtain file paths
    files = []
    for arg in args:
        path = Path(arg)
        try:  # check path exists
            assert path.exists()
        except AssertionError:
            print(f'Path: {path} does not exist. Skipping.')
            continue
        if path.is_dir():  # if path is a directory, search this directory and all sub-directories for csv files
            for f in path.glob('**/*.csv'):
                files.append(f)
        elif path.suffix == '.csv':  # if path is a csv file, add it to the list of files
            files.append(path)
        else:
            print(f'Path: {path} is not a valid csv file. Skipping.')
    # Open all valid csv files
    print(f'Opening {len(files)} csv files...')
    dfs = []
    indexer = []
    progress = PrintProgress(len(files))
    for i, f in enumerate(files):
        parts = [f.stem] + list(reversed(f.parent.parts))  # obtain parts of the path
        df = pd.read_csv(f)  # open the DataFrame
        dfs.append(df)  # append DataFrame to list of DataFrames
        indexer.append(tuple([parts[i] for i in index_levels]))  # add parts of the file path specified by the index
        # levels to the indexer
        progress(i)
    concatenated = pd.concat(dfs, keys=indexer, names=level_names)  # concatenate DataFrames
    print('done!')
    print(f'DataFrame has shape: {concatenated.shape}.\n'
          f'Index levels: {level_names}.\n')
    return concatenated


class PrintProgress:

    def __init__(self, n):
        self.n = n
        self.deci = int(self.n / 10)
        self.use_deci = (self.n >= 10)
        self.centi = int(self.n / 100)
        self.use_centi = (self.n >= 100)

    def __call__(self, i):
        if i > 0:
            if self.use_centi:
                if (i % self.deci) == 0:
                    print(f'# {(i // self.deci) * 10}%')
                elif (i % self.centi) == 0:
                    print('#', end='')
            elif self.use_deci and ((i % self.deci) == 0):
                print('#', end='')
            elif i == (self.n - 1):
                print('########## 100%')
