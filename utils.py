import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm.auto import tqdm

features_to_drop = ['instance weight',
                    'enroll in edu inst last wk',
                    'member of a labor union',
                    'reason for unemployment',
                    'region of previous residence',
                    'state of previous residence',
                    'migration prev res in sunbelt',
                    'family members under 18',
                    "fill inc questionnaire for veteran's admin"]

categorical_features = ['class of worker',
                        'detailed industry recode',
                        'detailed occupation recode',
                        'education',
                        'marital status',
                        'major industry code',
                        'major occupation code',
                        'race',
                        'hispanic origin',
                        'sex',
                        'full or part time employment stat',
                        'tax filer stat',
                        'detailed household and family stat',
                        'detailed household summary in household',
                        'migration code-change in msa',
                        'migration code-change in reg',
                        'migration code-move within reg',
                        'live in this house 1 year ago',
                        'country of birth father',
                        'country of birth mother',
                        'country of birth self',
                        'citizenship',
                        'own business or self employed',
                        'veterans benefits',
                        'year']

numerical_features = ['age',
                      'wage per hour',
                      'capital gains',
                      'capital losses',
                      'dividends from stocks',
                      'num persons worked for employer',
                      'weeks worked in year']


def reduce_mem_usage(df):
    start_mem_usg = df.memory_usage().sum() / 1024 ** 2

    print("Memory usage of dataframe: ", start_mem_usg, " MB")

    na_list = []  # Keeps track of columns that have missing values filled in.
    for col in tqdm(df.columns):
        if df[col].dtype != object:  # Exclude strings

            # make variables for Int, max and min
            is_int = False
            mx = df[col].max()
            mn = df[col].min()

            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(df[col]).all():
                na_list.append(col)
                df[col].fillna(mn - 1, inplace=True)

                # test if column can be converted to an integer
            asint = df[col].fillna(0).astype(np.int64)
            result = (df[col] - asint)
            result = result.sum()
            if -0.01 < result < 0.01:
                is_int = True

            # Make Integer/unsigned Integer datatypes
            if is_int:
                if mn >= 0:
                    if mx < 255:
                        df[col] = df[col].astype(np.uint8)
                    elif mx < 65535:
                        df[col] = df[col].astype(np.uint16)
                    elif mx < 4294967295:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)

                        # Make float datatypes 32 bit
            else:
                df[col] = df[col].astype(np.float32)

    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = df.memory_usage().sum() / 1024 ** 2
    print(f'Memory usage is: {mem_usg} MB')
    print(f'This is {100 * mem_usg / start_mem_usg:.2f}% of the initial size')

    return df, na_list


def nans_count(df, axis=0):
    nans_number = pd.DataFrame(df.isnull().sum(axis=axis) * 100 / df.shape[axis], columns=['nan_count'])
    return nans_number[nans_number['nan_count'] > 0]


def unique_values(df):
    for column in tqdm(df.columns):
        values = list(df[column].unique())
        print(f'{column}: {values}')


def pca_results(full_dataset, pca, show_plot=True):
    """
    Create a DataFrame of the PCA results
    Includes dimension feature weights and explained variance
    Visualizes the PCA results
    """

    # Dimension indexing
    dimensions = ['Dimension {}'.format(i) for i in range(1, len(pca.components_) + 1)]

    # PCA components
    components = pd.DataFrame(np.round(pca.components_, 4), columns=full_dataset.keys())
    components.index = dimensions

    # PCA explained variance
    ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
    variance_ratios = pd.DataFrame(np.round(ratios, 4), columns=['Explained Variance Ratio'])
    variance_ratios.index = dimensions

    # PCA explained cumulative variance
    cumsum = pca.explained_variance_ratio_.cumsum().reshape(len(pca.components_), 1)
    variance_cumsum = pd.DataFrame(np.round(cumsum, 4), columns=['Explained Cumulative Variance'])
    variance_cumsum.index = dimensions

    if show_plot:
        # Create a bar plot visualization
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.plot(np.arange(len(variance_cumsum)), variance_cumsum)
        ax.set_ylabel("Explained Cumulative Variance")
        ax.set_xlabel("Number of Principal Components")

    # Return a concatenated DataFrame
    return pd.concat([variance_cumsum, variance_ratios, components], axis=1)


# Map weights for the first principal component to corresponding feature names
# and then print the linked values, sorted by weight.
def print_pcs(df, pca, comp, k=5):
    components = pd.DataFrame(np.round(pca.components_, 4), columns=df.columns)
    pc = components.iloc[comp - 1].sort_values(ascending=False)
    print(f'Weights for PC{comp}')
    print(f'Top {k} weights')
    print(pc.head(k))
    print('\n')
    print(f'Bottom {k} weights')
    print(pc.tail(k))


def class_distribution(df):
    plt.figure(figsize=(5, 5))
    total = df.shape[0]
    ax = sns.countplot(x="income class", data=df)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2.,
                height + 3,
                '{0:.2%}'.format(height/total),
                ha="center")
    plt.title('Distribution of income classes', fontsize=15)
    plt.show()


def clean_dataset(df):
    # replace field that contains Not in Universe with NaN
    df = df.replace(r'Not in universe\w*?', np.nan, regex=True)

    # Drop features
    df = df.drop(features_to_drop, axis=1)

    # Re-encode features
    #df['sex'] = df['sex'].map({'Female': 0, 'Male': 1})
    df['income class'] = df['income class'].map({'- 50000.': 0, '50000+.': 1})
    #df = df.fillna('Unknown')
    #df[categorical_features] = df[categorical_features].astype(str)

    # Reduce memory usage
    df, _ = reduce_mem_usage(df)

    return df
