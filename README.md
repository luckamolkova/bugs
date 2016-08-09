# Bug Triaging

Data Science Project for [Galvanize](http://www.galvanize.com/), Seattle, August 2016
Lucie Klimosova ([LinkedIn](https://www.linkedin.com/in/lucieklimosova))

Hundreds of bug reports for large software projects are created daily.
Many of them contain incomplete, wrong or duplicate information.
Bug reporters as well as developers would benefit from "smarter" bug reporting system.

## Data

The Eclipse and Mozilla defect tracking dataset... [1]
- Mozilla subset (mostly Firefox, Thunderbird, Mozilla projects).
- Almost 400,000 bug reports resolved between 1998 and 2013.
- Additional information obtained from Bugzilla API - full description, id of duplicate.

## Insights

There are many areas for improvement.
72% of severity normal, 8% changes over the lifetime.
98% of priority empty, only 9% filled in over the lifetime.
Similar change rate applies to product, component and version.
The bug is assigned to many developers during the lifetime.

26% bug reports end up being closed as duplicates.

I have focused on predicting severity and priority and detecting duplicates.

## Modeling

### Severity and Priority

Default values (normal severity, empty priority) treated as unlabeled.
TfIdf on short descritption
TfIdf on long description
Gradient Boosting on other features
Linear stacking of resutls

### Duplicates
Problem on pairs of observations
Return duplicate high in the search
Similarity of description and product
(Experimented with LSH, doc2vec)

## Results

Better than or comparable with previously published researches. [2][3][4]

## Severity

class | precision  |  recall | f1-score |  support
---|---|---|---|---
    blocker   |    0.74  |   0.29  |   0.42   |   957
   critical   |    0.80  |   0.78  |   0.79   |  8816
      major   |    0.61  |   0.77  |   0.68   |  8770
      minor   |    0.59  |   0.47  |   0.52   |  4437
    trivial   |    0.68  |   0.43  |   0.53   |  1827
**avg / total**   |   **0.68**   |  **0.68**   |  **0.67**    | 24807

## Priority

class | precision  |  recall | f1-score |  support
---|---|---|---|---
         p1  |    0.61  |   0.74  |   0.67   |  2449
         p2  |    0.52  |   0.63  |   0.57   |  2378
         p3  |    0.57  |   0.41  |   0.48   |  1497
         p4  |    0.70  |   0.18  |   0.28   |   574
         p5  |    0.70  |   0.25  |   0.37   |   250
**avg / total**  |    **0.58**  |   **0.57**  |   **0.55**   |  7148

## Duplicates

top 3: 10%
top 10: 15%
top 50: 26%
7,000 candidates on average

**References**

[1] Lamkanfi, A., Pérez, J. and Demeyer, S., 2013, May. The eclipse and mozilla defect tracking dataset: a genuine dataset for mining bug information. In Proceedings of the 10th Working Conference on Mining Software Repositories (pp. 203-206). IEEE Press.
[2] Tian, Y., Lo, D. and Sun, C., 2012, October. Information retrieval based nearest neighbor classification for fine-grained bug severity prediction. In 2012 19th Working Conference on Reverse Engineering (pp. 215-224). IEEE.
[3] Lamkanfi, A., Demeyer, S., Giger, E. and Goethals, B., 2010, May. Predicting the severity of a reported bug. In 2010 7th IEEE Working Conference on Mining Software Repositories (MSR 2010) (pp. 1-10). IEEE.
[4] Menzies, T. and Marcus, A., 2008, September. Automated severity assessment of software defect reports. In Software Maintenance, 2008. ICSM 2008. IEEE International Conference on (pp. 346-355). IEEE.