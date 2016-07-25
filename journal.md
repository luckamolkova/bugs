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

