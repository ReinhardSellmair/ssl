# split and feature engineer data
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from config import TRAIN_SIZE, VAL_SIZE, RANDOM_STATE
from defs import TRAIN, VAL, TEST


def assign_data_set(df: pd.DataFrame, train_size: float=TRAIN_SIZE, val_size: float=VAL_SIZE, 
                    random_state: int=RANDOM_STATE) -> pd.DataFrame:
    """
    Shuffle data and assign training and validation data sets.
    @param df: data
    @param train_size: proportion of training data
    @param val_size: proportion of validation data
    @param random_state: random seed
    @return: data with data set assigned as index
    """
    
    # randomly shuffle data
    df = df.copy().sample(frac=1, random_state=random_state)

    train_samples = round(len(df) * train_size)
    val_samples = round(len(df) * val_size)

    # assign data set
    df['data_set'] = TEST
    df['data_set'].values[:train_samples] = TRAIN
    df['data_set'].values[train_samples:train_samples+val_samples] = VAL

    df = df.set_index('data_set', drop=True)

    return df


def ohe_features(df: pd.DataFrame, encode_cols: list[str]) -> pd.DataFrame:
    """
    One-hot encode features and append to DataFrame.
    @param df: data
    @param encode_cols: columns to one-hot encode
    @return: data with one-hot encoded features
    """
    encoder = OneHotEncoder(sparse_output=False)

    # fit encoder
    _ = encoder.fit(df.loc[TRAIN, encode_cols])

    # transform the data
    encoded = encoder.transform(df[encode_cols])

    # Get feature names
    feature_names_out = encoder.get_feature_names_out(encode_cols)

    # Create a new DataFrame with the one-hot encoded variables
    encoded_df = pd.DataFrame(encoded, columns=feature_names_out, index=df.index)

    # Concatenate the original DataFrame with the one-hot encoded DataFrame
    return pd.concat([df.drop(columns=encode_cols), encoded_df], axis=1)
