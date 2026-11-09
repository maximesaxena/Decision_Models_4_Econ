# SCE Household Spending Survey human data

From the New York Fed's [Survey of Consumer Expectations](https://www.newyorkfed.org/microeconomics/sce)
(SCE), © 2013-26 Federal Reserve Bank of New York, used subject to the
[license terms](https://www.newyorkfed.org/microeconomics/sce) posted there. The SCE is a monthly online
survey of a rotating panel of about 1,300 household heads, each answering for up to 12 months.

| file | contents |
|---|---|
| sce-household-spending-microdata.xlsx | the [Household Spending Survey](https://www.newyorkfed.org/microeconomics/sce/household-spending), a module asked every 4 months (April, August, December) to that month's SCE panel: Dec 2014 to Dec 2024, 33,782 answers from 15,052 people, keyed by `userid` and `date` (YYYYMM). No demographics or weights. |
| sce_household-spending-questionnaire.pdf | its questionnaire |
| frbny-sce-public-microdata-20-24.xlsx | the core monthly SCE, 2020-2024: weights (`weight`), demographics (Q32-Q47, asked in a person's first month only; the Fed's `_AGE_CAT`, `_EDU_CAT`, `_HH_INC_CAT` bands in every month), employment (Q10) and state (`_STATE`). The core questionnaire is [on the SCE page](https://www.newyorkfed.org/medialibrary/interactives/sce/sce/downloads/data/frbny-sce-survey-core-module-public-questionnaire.pdf). |
| respondents.csv | built by `src/experiments/sce_spending/data.py`: one row per spending answer from 2020 on, with that person's demographics and weight |

Only spending answers from 2020 on can be matched to core answers here (all of them are, by `userid` and
`date`); covering 2014-19 needs the core files for 2013-16 and 2017-19 from the SCE page. People who
entered the panel in 2019 have no demographics here, which thins out the 2020 waves.

The experiment uses two questions, QSP12n (what you would do with 10% more household income) and
QSP13new (10% less), each with 7 options; respondents choosing a mix (options 4-7) then split it in
QSP12a / QSP13a. In 2020-24, about 20% of a 10% gain would be spent, while about 72% of a 10% loss would
be covered by cutting spending.
