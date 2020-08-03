"""
Read/Write functions used for data in Kipet
"""
from contextlib import contextmanager
import inspect
import os
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

DEFAULT_DIR = 'data_sets'

# def set_data_file_path(filename, directory='data_sets'):
    
#     print(inspect.getfile(inspect.currentframe())) # script filename (usually with path)
#     print(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))) # script directory
    
#     dataDirectory = os.path.abspath(
#     os.path.join( os.path.dirname( os.path.abspath( inspect.getfile(
#         inspect.currentframe() ) ) ), directory))
#     filename =  os.path.join(dataDirectory, filename)
    
#     return filename

def _set_directory(filename, directory=DEFAULT_DIR):
    """Sets the current working directory plus the given subdirectory as the
    directory for the given data file (filename)
    
    Args:
        filename (str): name of input file
        
        directory (str): name of directory with data file
          
    Returns:
        filename (Path): the full filename of the data
    
    """
    #target_filename = inspect.getframeinfo(inspect.currentframe()).filename
    #path = os.path.dirname(os.path.abspath(target_filename))
    #current_dir = Path(path)    
    #kipet_dir = current_dir.parent.parent
    #data_dir = kipet_dir.joinpath(directory)
    #abs_filename = data_dir.joinpath(Path(filename))

    kipet_dir = Path.cwd()
    data_dir = kipet_dir.joinpath(directory)
    abs_filename = data_dir.joinpath(Path(filename))
    return abs_filename

def read_file(filename, directory=DEFAULT_DIR):       
    """ Reads data from a csv or txt file and converts it to a DataFrame
    
        Args:
            filename (str): name of input file
            
            directory (str): name of data directory if not Kipet default
          
        Returns:
            DataFrame

    """
    filename = _set_directory(filename, directory)
    print(f'read dir : {filename}')
    #filename = Path(filename)
    data_dict = {}
    if filename.suffix == '.txt':
    
        with open(filename, 'r') as f:
            for line in f:
                if line not in ['','\n','\t','\t\n']:
                    l = line.split()
                    if is_float_re(l[1]):
                        l[1] = float(l[1])
                    data_dict[float(l[0]), l[1]] = float(l[2])
        
        df_data = dict_to_df(data_dict)
        df_data.sort_index(ascending=True, inplace=True)
        return df_data

    elif filename.suffix == '.csv':
        
        df_data = pd.read_csv(filename, index_col=0)
        return df_data   

    else:
        raise ValueError(f'The file extension {filename.suffix} is currently not supported')
        return None

def write_file(filename, dataframe, filetype='csv', directory=DEFAULT_DIR):
    """ Write data to file.
    
        Args:
            filename (str): name of output file
          
            dataframe (DataFrame): pandas DataFrame
        
            filetype (str): choice of output (csv, txt)
        
        Returns:
            None

    """
    if filetype not in ['csv', 'txt']:
        print('Savings as CSV - invalid file extension given')
        filetype = 'csv'
    
    suffix = '.' + filetype
    
    filename = Path(directory).joinpath(filename)
    if filename.suffix == '':
        filename = filename.with_suffix(suffix)
    else:
        suffix = filename.suffix
        if suffix not in ['.txt', '.csv']:
            print('Savings as CSV - invalid file extension given')
            filename = Path(filename.stem).with_suffix('.csv')
    
    if filename.suffix == '.csv':
        dataframe.to_csv(filename)

    elif filename.suffix == 'txt':    
        with open(filename, 'w') as f:
            for i in dataframe.index:
                for j in dataframe.columns:
                    if not np.isnan(dataframe[j][i]):
                        f.write("{0} {1} {2}\n".format(i,j,dataframe[j][i]))
                        
    print(f'Data successfully saved as {filename}')
    return None

def read_spectral_data_from_csv(filename, instrument = False, negatives_to_zero = False):
    """ Reads csv with spectral data
    
        Args:
            filename (str): name of input file
            instrument (bool): if data is direct from instrument
            negatives_to_zero (bool): if data contains negatives and baseline shift is not
                                        done then this forces negative values to zero.

        Returns:
            DataFrame

    """
    data = pd.read_csv(filename,index_col=0)
    if instrument:
        #this means we probably have a date/timestamp on the columns
        data = pd.read_csv(filename,index_col=0, parse_dates = True)
        data = data.T
        for n in data.index:
            h,m,s = n.split(':')
            sec = (float(h)*60+float(m))*60+float(s)
            data.rename(index={n:sec}, inplace=True)
        data.index = [float(n) for n in data.index]
    else:
        data.columns = [float(n) for n in data.columns]

    #If we have negative values then this makes them equal to zero
    if negatives_to_zero:
        for t in (data.index):
            for l in data.columns:
                if data.loc[t,l] < 0:
                    data.loc[t,l] = 0.0

    return data

# Legacy functions

def write_spectral_data_to_csv(filename, dataframe, directory=DEFAULT_DIR):
    """
        Args:
            filename (str): name of output file
          
            dataframe (DataFrame): pandas DataFrame
        
        Returns:
            None
    """
    write_file(filename, dataframe, 'csv', directory)

def write_spectral_data_to_txt(filename, dataframe, directory=DEFAULT_DIR):
    """ Write spectral data Dij to txt file.
    
        Args:
            filename (str): name of output file
          
            dataframe (DataFrame): pandas DataFrame
        
        Returns:
            None
    
    """
    write_file(filename, dataframe, 'txt', directory)

def write_absorption_data_to_csv(filename, dataframe, directory=DEFAULT_DIR):
    """ Write absorption data Sij to csv file.
    
        Args:
            filename (str): name of output file
          
            dataframe (DataFrame): pandas DataFrame
        
        Returns:
            None
    """
    write_file(filename, dataframe, 'csv', directory)

def write_absorption_data_to_txt(filename, dataframe, directory=DEFAULT_DIR):
    """ Write absorption data Sij to txt file.
    
        Args:
            filename (str): name of output file
          
            dataframe (DataFrame): pandas DataFrame
        
        Returns:
            None
    """
    write_file(filename, dataframe, 'txt', directory)

def write_concentration_data_to_csv(filename, dataframe, directory=DEFAULT_DIR):
    """ Write concentration data Cij to csv file.
    
        Args:
            filename (str): name of output file
          
            dataframe (DataFrame): pandas DataFrame
        
        Returns:
            None
    """
    write_file(filename, dataframe, 'csv', directory)

def write_concentration_data_to_txt(filename, dataframe, directory=DEFAULT_DIR):
    """ Write concentration data Cij to txt file.
    
        Args:
            filename (str): name of output file
          
            dataframe (DataFrame): pandas DataFrame
        
        Returns:
            None
    """
    write_file(filename, dataframe, 'txt', directory)


def read_concentration_data(filename):
    
    print(filename)
    if filename.suffix == '.csv':
        return read_concentration_data_from_csv(filename)
    elif filename.suffix == '.txt':
        return read_concentration_data_from_txt(filename)
    else:
        raise ValueError('Filetype not csv or txt.')
        return None

def read_concentration_data_from_txt(filename, directory=DEFAULT_DIR):
    """ Reads txt with concentration data
    
        Args:
            filename (str): name of input file
          
        Returns:
            DataFrame
    """
    return read_file(filename, directory)

def read_concentration_data_from_csv(filename, directory=DEFAULT_DIR):
    """ Reads csv with concentration data
    
        Args:
            filename (str): name of input file
         
        Returns:
            DataFrame
    """
    return read_file(filename, directory)


def read_absorption_data_from_csv(filename, directory=DEFAULT_DIR):
    """ Reads csv with spectral data
    
        Args:
            filename (str): name of input file
          
        Returns:
            DataFrame
    """
    return read_file(filename, directory)


def read_spectral_data_from_txt(filename, directory=DEFAULT_DIR):
    """ Reads txt with spectral data
    
        Args:
            filename (str): name of input file
          
        Returns:
            DataFrame
    """
    return read_file(filename, directory)


def read_absorption_data_from_txt(filename, directory=DEFAULT_DIR):
    """ Reads txt with absorption data
    
        Args:
            filename (str): name of input file
          
        Returns:
            DataFrame
    """
    return read_file(filename, directory)


# for redirecting stdout to files
@contextmanager
def stdout_redirector(stream):
    old_stdout = sys.stdout
    sys.stdout = stream
    try:
        yield
    finally:
        sys.stdout = old_stdout
    
# Data conversion tools

def dict_to_df(data_dict):

    """Takes a dictionary of typical pyomo data and converts it to a dataframe
    
    """    
    dfs_stacked = pd.Series(index=data_dict.keys(), data=list(data_dict.values()))
    dfs = dfs_stacked.unstack()
    return dfs

def is_float_re(str):
    """Checks if a value is a float or not"""
    _float_regexp = re.compile(r"^[-+]?(?:\b[0-9]+(?:\.[0-9]*)?|\.[0-9]+\b)(?:[eE][-+]?[0-9]+\b)?$").match
    return True if _float_regexp(str) else False

def df_from_pyomo_data(varobject):

    val = []
    ix = []
    for index in varobject:
        ix.append(index)
        val_raw = varobject[index].value
        if val_raw is None:
            val_raw = 0
        val.append(val_raw)
    
    a = pd.Series(index=ix, data=val)
    dfs = pd.DataFrame(a)
    index = pd.MultiIndex.from_tuples(dfs.index)
   
    dfs = dfs.reindex(index)
    dfs = dfs.unstack()
    dfs.columns = [v[1] for v in dfs.columns]

    return dfs

