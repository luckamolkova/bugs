# Bug Triaging

- Data Science Project for [Galvanize](http://www.galvanize.com/), Seattle, August 2016.
- Lucie Klimosova ([LinkedIn](https://www.linkedin.com/in/lucieklimosova)).
- See the web application live on [bugtriaging.com](http://bugtriaging.com).

**Motivation**

Large software projects can produce hundreds of bug reports every day. Many of these reports contain incomplete, incorrect, or redundant information.

Bug reporters and developers would benefit from a "smarter" bug reporting system. This projects shows that severity and priority predictions can be assigned to tickets automatically. It also attempts to prevent duplicate creation by suggesting similar bug reports.

## Data

The Eclipse and Mozilla defect tracking dataset... [1]
- Mozilla subset (mostly Firefox, Thunderbird and Mozilla projects).
- Almost 400,000 bug reports resolved between 1998 and 2013.
- Additional information obtained from [Bugzilla API](https://wiki.mozilla.org/Bugzilla:REST_API)
  - Full description - stored as first comment ([API call](https://bugzilla.mozilla.org/rest/bug/707428/comment))
  - ID of duplicate ([API call](https://bugzilla.mozilla.org/rest/bug/76103?include_fields=dupe_of))

The bug report is provided as a series of incremental updates appearing during its lifetime. Thus, the report can be analyzed both at the time of creation and at the time of closure.

## Insights

There are many areas for improvement.
- 72% of severity normal (default), 8% changes over the lifetime.
- 98% of priority empty (default), only 9% filled in over the lifetime.
- Similar change rate applies to product, component and version.
- The bug is assigned to many developers during the lifetime.
- 26% bug reports end up being closed as duplicates.

## Modeling and Results

### Severity and Priority

Default values (normal severity, empty priority) were treated as unlabeled.

Three different models were trained:

1. Multinomial Naive Bayes on TfIdf of short descritption (250 characters).
2. Multinomial Naive Bayes on TfIdf of long description.
3. Gradient Boosting on other features, including
  - "Skill" of the reporter - how many bug reports she has created before.
  - How many people are on the cc list.
  - Is the bug report assigned immediately.
  - etc.

Linear stacking was used to combine the probability predictions of the three models:

- Gradient Boosting on predicted probabilities for each class.

**Severity Results**

Better than or comparable with previously published researches. [2][3][4]

class | precision  |  recall | f1-score |  support
---|---|---|---|---
    blocker   |    0.74  |   0.29  |   0.42   |   957
   critical   |    0.80  |   0.78  |   0.79   |  8816
      major   |    0.61  |   0.77  |   0.68   |  8770
      minor   |    0.59  |   0.47  |   0.52   |  4437
    trivial   |    0.68  |   0.43  |   0.53   |  1827
**avg / total**   |   **0.68**   |  **0.68**   |  **0.67**    | 24807

**Priority Results**

class | precision  |  recall | f1-score |  support
---|---|---|---|---
         p1  |    0.61  |   0.74  |   0.67   |  2449
         p2  |    0.52  |   0.63  |   0.57   |  2378
         p3  |    0.57  |   0.41  |   0.48   |  1497
         p4  |    0.70  |   0.18  |   0.28   |   574
         p5  |    0.70  |   0.25  |   0.37   |   250
**avg / total**  |    **0.58**  |   **0.57**  |   **0.55**   |  7148

### Duplicates

This problem exists for pairs of observations rather than for a single observation.
- Typically highly imbalanced - each bug report has one or two duplicates and there are thousands or more non-duplicates for every set of duplicates.
- The goal is to return the actual duplicate as high in the search as possible.

Assumptions that help reduce the number of potential candidates searched:
- The original should exist and should not be resolved at the time of duplicate creation.
- Both the duplicate and the original are reported for the same product.

Features used:
- Cosine similarity of TfIdf of short descritption (250 characters).
- Cosine similarity of TfIdf of long descritption.
- Similarity of component and operating system.
- Reporter (the assumption is that the same person is not likely to report the same thing twice).

I have also experimented with locality sensitive hashing (LSH) implmented in [NearPy](https://github.com/pixelogik/NearPy) and [doc2vec](https://radimrehurek.com/gensim/models/doc2vec.html) and [text summarization](http://rare-technologies.com/text-summarization-with-gensim/) implemented in gensim to see if it could be used instead of simple cosine similarity on TfIdf matrices but my results were not better.

**Duplicate Results**

- Actual duplicate returned in top 3: **10%**
- Actual duplicate returned in top 10: **15%**
- Actual duplicate returned in top 50: **26%**

The results are not impressive but it is important to keep in mind that there were 7,000 candidates considered on average. The model would perform much better on smaller set of observations (hundreds to thousands of bug reports).

## Deployment

To learn more about the code and how to use it, look at [technical documentation](src/technical_documentation.md).

## Next Steps

Improve existing predictions:

- Explore more features available in the API (newer bug reports have information including keywords or votes).
- Improve the duplicate search, spend more time on text similarity detection.

Make the bug triaging solution more complete by adding more functionality:

- Predict component and product.
- Suggest assignees.
- Predict resolution time.

**References**

[1] Lamkanfi, A., Pérez, J. and Demeyer, S., 2013, May. The eclipse and mozilla defect tracking dataset: a genuine dataset for mining bug information. In Proceedings of the 10th Working Conference on Mining Software Repositories (pp. 203-206). IEEE Press.

[2] Tian, Y., Lo, D. and Sun, C., 2012, October. Information retrieval based nearest neighbor classification for fine-grained bug severity prediction. In 2012 19th Working Conference on Reverse Engineering (pp. 215-224). IEEE.

[3] Lamkanfi, A., Demeyer, S., Giger, E. and Goethals, B., 2010, May. Predicting the severity of a reported bug. In 2010 7th IEEE Working Conference on Mining Software Repositories (MSR 2010) (pp. 1-10). IEEE.

[4] Menzies, T. and Marcus, A., 2008, September. Automated severity assessment of software defect reports. In Software Maintenance, 2008. ICSM 2008. IEEE International Conference on (pp. 346-355). IEEE.
