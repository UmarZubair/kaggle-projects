def preprocess_df(df):
    print(df.head())
    print(df.columns)
    print(df.isnull().sum())
    print(df['keyword'].unique().size)
    return df