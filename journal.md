## Getting data

### Setting up PostgreSQL Database

- Create new database.

```
$ psql
# create database bugs
```

- Fill in database connection details (name, username, host) in `util.py`.

### Moving Data from JSON to PostgreSQL

- Download the `v02` of Mozilla and Eclipse Defect Tracking Dataset from [GitHub](https://github.com/ansymo/msr2013-bug_dataset)
- Set path to the dataset in in `get_json_data.py`.
- Run `get_json_data.py` from terminal:

```
$ time python get_json_data.py
```

The code runs for roughly 40 minutes:

```
> 19:22:15: processing /msr2013-bug_dataset/data/v02/mozilla/reports.json...
> 19:29:09: processing /msr2013-bug_dataset/data/v02/mozilla/cc.json...
> 19:37:45: processing /msr2013-bug_dataset/data/v02/mozilla/assigned_to.json...
> 19:40:54: processing /msr2013-bug_dataset/data/v02/mozilla/bug_status.json...
> 19:46:17: processing /msr2013-bug_dataset/data/v02/mozilla/component.json...
> 19:49:01: processing /msr2013-bug_dataset/data/v02/mozilla/op_sys.json...
> 19:51:24: processing /msr2013-bug_dataset/data/v02/mozilla/priority.json...
> 19:53:41: processing /msr2013-bug_dataset/data/v02/mozilla/product.json...
> 19:56:00: processing /msr2013-bug_dataset/data/v02/mozilla/resolution.json...
> 20:00:12: processing /msr2013-bug_dataset/data/v02/mozilla/severity.json...
> 20:02:31: processing /msr2013-bug_dataset/data/v02/mozilla/short_desc.json...
> 20:05:08: processing /msr2013-bug_dataset/data/v02/mozilla/version.json...
```

### Getting Bug Report Descriptions

- Run `get_descriptions.py` from terminal as many times as needed (it gets 100,000 descriptions in one go):

```
$ time python get_descriptions.py
```

Bugzilla stores long description as a first comment to the bug report.

The code goes through all bug reports in POstgreSQL database that do not have description yet and tries to get it from bugzilla.mozilla.org API (request is formatted as https://bugzilla.mozilla.org/rest/bug/707428/comment).

The first comment, its time and bug_id are stored in table called `descriptions`. Threading is used to speed up the process. Data is commited after every 100 descriptions (roughly 15 seconds, depending on internet connection speed).

```
> 10:50:56 inserted 100 rows
> 10:51:12 inserted 100 rows
> 10:51:26 inserted 100 rows
> 10:51:42 inserted 100 rows
> 10:51:58 inserted 100 rows
```

Overall there are roughly 400,000 bug reports; it should take roughly 17 hours to get all descriptions (`get_descriptions.py` needs to be triggered 4 times as it is set to get 100,000 at a time).

16 bugs are hardcoded to be skipped because they either have no description or they are not publicly available:

```
821596, 627634, 525063, 661158, 758675, 773487, 808808, 804674, 724355, 795589, 808857, 574459
```

## Data Preprocessing

- Run `data_preprocessing.py` from terminal:

```
$ time python data_preprocessing.py
```

The code runs roughly 3 minutes. Table named `final` is created in the database.

```
> 15:31:10: creating base table...
> 15:31:12: creating assigned_to_clean table...
> 15:31:13: creating assigned_to_2 temp table...
> ...
> 15:33:18: creating version_clean table...
> 15:33:19: creating version_2 temp table...
> 15:33:26: creating final table...
```

## Data Exploration

- 310,947 bugs in total
- opening: from 1996-03-11 to 2013-01-01
- closing: from 1998-08-27 to 2013-01-01

### Reassignments

```
priority:    6 unique values, 27,319  (8.79%) bugs reassigned
severity:    7 unique values, 26,028  (8.37%) bugs reassigned
component: 730 unique values, 67,343 (21.66%) bugs reassigned
product:    69 unique values, 24,568  (7.90%) bugs reassigned
```

**Priority** - Most of the time empty, not easily predictable.

**Severity** - Worth trying.

**Component** - Too many unique components, not easily predictable.

**Product** - 69 unique values is still a little bit too much. Try to keep top 7 products ('core', 'firefox', 'thunderbird', 'bugzilla', 'browser', 'webtools', 'psm') and 'other'?

```
product:     8 unique values, 25,563  (7.89%) bugs reassigned
```

### Duplicates

26% bug reports end up being closed as duplicates:

```
fixed         110,143
duplicate      82,770
worksforme     47,813
invalid        34,219
incomplete     21,548
wontfix         9,843
expired         4,593
moved              14
                    4
```

### Resolution Time

- Calculated as `reports.opening` - last(`resolution.when`)

```
mean  206 days 18:34:58.243462
std   451 days 15:06:01.450312    
min     0 days 00:00:00 
25%     0 days 18:38:06.500000  
50%    16 days 12:30:30    
75%   183 days 09:30:08.500000    
max  4767 days 04:37:06
```

- We can split the data into buckets based on number of days as follows:

```
day     0-1 days  14,804
week    1-7 days  35,255
month  7-30 days  40,117
year 30-365 days  82,785
more   365+ days  53,125
```

## Data Preparation

- Only keep stuff with description
- Only keep closed (././.)
- Get rid of enhancements (Bugzilla allows users to request new features, which technically do not represent real bugs. Therefore, we do not consider reports where the severity attribute is set to enhancement as this category is reserved for feature requests.)
- Calculate resolution time in days as `reports.opening` - last(`resolution.when`)

## Modeling

### Predict Resolution Time using description

Only look at fixed
```
df_fixed = df[df['resolution_final'] == 'fixed']
```

**Classifier**

```
df_fixed['duration_bin2'] = pd.qcut(df_fixed['duration_days'], 5)
df_fixed['duration_bin2'].value_counts()
[0, 2]         25791
(123, 4567]    21991
(30, 123]      21872
(9, 30]        20664
(2, 9]         19825
```

Majority is 23.4060067549% cases in train
Majority is 23.4456711214% cases in test

```
nb_model = MultinomialNB()
accuracy: 0.295177222545
precision: 0.267809211621
recall: 0.295177222545
confusion matrix: 
 [[2665   53  671  154 2028]
 [1214   75  547  165 2965]
 [1936   80  823  183 2408]
 [1474   68  632  207 2732]
 [1352   70  535  141 4358]]

rf_model = RandomForestClassifier(n_estimators=20, criterion='gini', 
                               max_depth=3, max_features='auto', 
                               bootstrap=True, oob_score=True,
                               random_state=None, warm_start=False)
accuracy: 0.251670540383
precision: 0.170135880347
recall: 0.251670540383
confusion matrix: 
 [[ 742    0    8    3 4818]
 [ 258    0    2    1 4705]
 [ 474    1    2    0 4953]
 [ 323    0    7    1 4782]
 [ 259    0   10    2 6185]]

gb_model = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, 
                                   n_estimators=100, subsample=1.0,
                                   max_depth=3, init=None, 
                                   random_state=None, max_features=None, 
                                   verbose=0, max_leaf_nodes=None, warm_start=False)
accuracy: 0.289221382917
precision: 0.2719571918
recall: 0.289221382917
confusion matrix: 
 [[1957   41  811   78 2684]
 [ 769   76  602  101 3418]
 [1300   64  943  130 2993]
 [ 935   65  731  131 3251]
 [ 814   90  607   88 4857]]
```

**Regressor**

```
df_fixed['duration_days_log'] = df_fixed['duration_days'].map(lambda x: np.log10(x + 1))
df_fixed['duration_days_log'].hist()
```

```
rf1_model = RandomForestRegressor(n_estimators=20, criterion='mse', 
                               max_depth=3, max_features='auto', 
                               bootstrap=True, oob_score=True,
                               random_state=None, warm_start=False)
oob score: 0.0390567043731
r-squared: 0.0356680001073
```

### Predict Resolution Time using other features

**Classifier**

```
df_fixed['duration_bin2'] = pd.qcut(df_fixed['duration_days'], 5, labels=[0,1,2,3,4])
df_fixed['duration_bin2'].value_counts()
0    25791
4    21991
3    21872
2    20664
1    19825
```

Majority is 23.4060067549% cases in train
Majority is 23.4456711214% cases in test

```
nb_model = MultinomialNB()
accuracy: 0.262165891923
precision: 0.225583691589
recall: 0.262165891923
confusion matrix: 
 [[4833    6   10  346 1261]
 [3426    6   11  335 1188]
 [3344    6    8  360 1395]
 [3355    6    7  398 1664]
 [3148    6   11  432 1974]]

 rf_model = RandomForestClassifier(n_estimators=20, criterion='gini', 
                               max_depth=3, max_features='auto', 
                               bootstrap=True, oob_score=True,
                               random_state=None, warm_start=False)
accuracy: 0.287877687391
precision: 0.181562189074
recall: 0.287877687391
confusion matrix: 
 [[4709    1    0  972  774]
 [3222    0    0  984  760]
 [3042    0    0 1091  980]
 [2874    0    0 1234 1322]
 [2434    0    0 1153 1984]]

gb_model = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, 
                                   n_estimators=100, subsample=1.0,
                                   max_depth=3, init=None, 
                                   random_state=None, max_features=None, 
                                   verbose=0, max_leaf_nodes=None, warm_start=False)
accuracy: 0.297247239977
precision: 0.262224082459
recall: 0.297247239977
confusion matrix: 
 [[4150   83  110  885 1228]
 [2677   66  120  914 1189]
 [2520   54  127 1035 1377]
 [2197   57  137 1155 1884]
 [1707   44  107 1026 2687]]
```

**Regressor**

```
df_fixed['duration_days_log'] = df_fixed['duration_days'].map(lambda x: np.log10(x + 1))
df_fixed['duration_days_log'].hist()
```

```
rf_model = RandomForestRegressor(n_estimators=20, criterion='mse', 
                               max_depth=3, max_features='auto', 
                               bootstrap=True, oob_score=True,
                               random_state=None, warm_start=False)
oob score: 0.0734031807501
r-squared: 0.06963286351
```

## Ran Manually

```
CREATE TABLE IF NOT EXISTS reporter_bugs AS (
  SELECT 
    f1.id AS id
    , COUNT(f2.id) AS reporter_bug_cnt
  FROM final f1
    LEFT OUTER JOIN final f2
      ON f1.reporter = f2.reporter
      AND f1.opening > f2.opening
  GROUP BY f1.id
);
```