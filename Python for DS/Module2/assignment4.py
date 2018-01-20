import pandas as pd

# TODO: Load up the table, and extract the dataset
# out of it. If you're having issues with this, look
# carefully at the sample code provided in the reading
url = 'http://www.espn.com/nhl/statistics/player/_/stat/points/sort/points/year/2015/seasontype/2'
site = pd.read_html(url)
df = site[0]

# TODO: Rename the columns so that they are similar to the
# column definitions provided to you on the website.
# Be careful and don't accidentially use any names twice.
df.columns = df.loc[1,:]
df.drop(df.index[:2], inplace=True)

# TODO: Get rid of any row that has at least 4 NANs in it,
# e.g. that do not contain player points statistics
df.dropna(axis=0, thresh=4, inplace=True)

# TODO: At this point, look through your dataset by printing
# it. There probably still are some erroneous rows in there.
# What indexing command(s) can you use to select all rows
# EXCEPT those rows?


# TODO: Get rid of the 'RK' column
df.drop(axis=1, labels=['RK'], inplace=True)

# TODO: Ensure there are no holes in your index by resetting
# it. By the way, don't store the original index
df.reset_index(drop=True, inplace=True)

# TODO: Check the data type of all columns, and ensure those
# that should be numeric are numeric
numcols = [col for col in df.columns if col not in ['PLAYER', 'TEAM']]
for col in numcols:
    df[col] = df[col].apply(pd.to_numeric, errors='coerce')
df.dropna(axis=0, thresh=4, inplace=True)

# TODO: Your dataframe is now ready! Use the appropriate 
# commands to answer the questions on the course lab page.

