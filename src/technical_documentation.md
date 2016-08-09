# Bug Triaging - Technical Documentation

## Getting data

#### Setting up PostgreSQL Database

- Create new database.

```
$ psql
# create database bugs
```

- Fill in database connection details (name, username, host) in `util.py`.

#### Moving Data from JSON to PostgreSQL

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

#### Getting Bug Report Descriptions

- Run `get_descriptions.py` from terminal as many times as needed (it gets 100,000 descriptions in one go):

```
$ time python get_descriptions.py
```

Bugzilla stores long description as a first comment to the bug report.

The code goes through all bug reports in PostgreSQL database that do not have description yet and tries to get it from bugzilla.mozilla.org API (request is formatted as https://bugzilla.mozilla.org/rest/bug/707428/comment).

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

- Only keep bug reports where the long description was successfully recieved from Bugzilla API.
- Only keep "closed" bug reports - final status is Resolved, Closed or Verified.
- Get rid of enhancements. (Bugzilla allows users to request new features, which technically do not represent real bugs. Therefore, we do not consider reports where the severity attribute is set to enhancement as this category is reserved for feature requests.)
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

#### Getting Bug Report "duplicate of"

- Run `get_duplicates.py` from terminal:

```
$ time python get_duplicates.py
```

The code goes through all bug reports in the final table that were closed as duplicates and tries to get `dupe_of` from bugzilla.mozilla.org API (request is formatted as https://bugzilla.mozilla.org/rest/bug/707428?include_fields=dupe_of).

The results are stored in table named `duplicates`.

## Data Exploration

- Code and detailed results are in `data_exploration.ipynb`.

- 310,947 bugs in total
- opening: from 1996-03-11 to 2013-01-01
- closing: from 1998-08-27 to 2013-01-01

**Reassignments**

```
priority:    6 unique values, 27,319  (8.79%) bugs reassigned
severity:    7 unique values, 26,028  (8.37%) bugs reassigned
component: 730 unique values, 67,343 (21.66%) bugs reassigned
product:    69 unique values, 24,568  (7.90%) bugs reassigned
```

**Product** - 69 unique values is still a little bit too much. Try to keep top 7 products ('core', 'firefox', 'thunderbird', 'bugzilla', 'browser', 'webtools', 'psm') and 'other'?

```
product:     8 unique values, 25,563  (7.89%) bugs reassigned
```

**Duplicates**

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

## Modeling

- Run `pickle_bug_pipelines.py` and `pickle_duplicate_pipeline.py` from terminal:

```
$ time python pickle_bug_pipelines.py
$ time python pickle_duplicate_pipeline.py
```

Three files should be pickled into data directory. All three of them are needed to run the web application:
- `severity_final_pipeline.pkl`
- `priority_final_pipeline.pkl`
- `duplicate_pipeline.pkl`

During the process of pickling, the pipelines are fitted, the models are trained (including TfIdf and Count vectorizers) and saved as part of the pickled pipeline. The pipelines then can be used for predictions in the web app. A few things that happen worth mentioning for specific models.

**Severity and Priority**

- Get rid of defaults (normal for severity, empty and -- for priority).
- Train test split.
- TfIdf vectorize short and long description.
- Count vectorize categorical attributes - product, component, operating system.
- Train three models in the first level using K-fold cross validation.
- Train the final model on probability predictions of previous model - using the data from K-fold validation that were not used for training the three models to avoid leakage.

**Duplicates**

- Only use bug reports that are known to be duplicates.
- Train test split.
- Generate pairs of observations. Pair each bug report with its real duplicate (target 1) and with 7 non-duplicates (label 0).
- Calculate cosine distances on TfIdf of short and long description, create features indicating whether the pair of bug reports has the same component, operating system and reporter.
- Train and pickle the model.
- Evaluation is optional. When you uncomment the line to evaluate the model, a table `duplicate_eval` will be created. For each bug report in the test set a record will be inserted with total nuber of candidates considered and with the position of actual duplicate amongst those candidates.

## Deployment

- Make sure the pickled pipelineas are in data directory.
- Start the flask app from terminal:

```
$ python app.py
```

- Access the app from `http://localhost:5353`.
- As of now (and for next few months) the app can be accessed from [bugtriaging.com](http://bugtriaging.com).