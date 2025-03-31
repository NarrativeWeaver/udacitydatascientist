## Palliative Care Analaysis

## Project Overview
 This project analyzes 3 palliative care metrics across 11 cancer hospitals in the US. It identifies relationships between these metrics, distributon across the centers and regions, and proposes potential interventions for improvement.
 
## Motivation
Palliative care is an essential form of care, which seeks to relieve patients of symptoms, pain, discomfort and suffering, associated with diseases such as cancer. For cancer patients, the transition from curative to palliative represents a crucial moment, with significant consequences for quality of life and dignity.

This project was motivated by a desire to understand how dfferent cancer centers perform on key metrics related to palliative care. By analyzing the data, we can identify patterns and relationships, which could present opportunities for improvement that could improve experience and quality of life of patients with terminal cancer.

## Data Description

The dataset (PCH_Palliative_Care_HOSPITAL.csv) contains information from 11 specialized cancer centers across 8 states, tracking four key end-of-life care metrics:

- PCH-32: Proportion of patients who died from cancer receiving chemotherapy in the last 14 days of life 
- PCH-33: Proportion of patients who died from cancer admitted to the ICU in the last 30 days of life 
- PCH-34: Proportion of patients who died from cancer not admitted to hospice 
- PCH-35: Proportion of patients who died from cancer admitted to hospice for less than 3 days 


The blog which contains an overview of the project can be found here: https://medium.com/@narrativeweaver/data-analysis-palliative-care-metrics-for-cancer-patients-across-11-us-hospitals-11ed5862f887

## Files

The repository includes the following files:
- `README(Data Science Blog Post).md`: Overview of data science project
- `Pallative Care Data Analysis (1 April 25)`: Jupyter Notebook containing the code and detailed analysis.
- `PCH_Palliative_Care_HOSPITAL.csv`: 4 palliative care metrics across 11 cancer centers in the US
- 'measure_distributions.png': Distribution of values for each metric
- 'correlation_matrix.png': Correlation analysis between metrics
- 'hospital_comparison'.png: Comparison of all hospitals across metrics
- 'state_comparison.png': Regional patterns in palliative care
- 'outlier_analysis.png': Analysis of hospitals with unusual patterns
- 'prediction_results.png': Model predictions vs. actual values
- 'simulation_results.png;: Results of intervention simulations
- 'hospital_ranking.png': Composite performance ranking

## Project Questions

The analysis explores the following business questions:

1. How do the 11 cancer centers compare in their end-of-life care metrics?
2. What are the relationships between the different palliative care metrics?
3. Can we make predictions of one metric from the others?
4. Is there a way of assessing the overall palliative care provided by the hospitals based on all the available metrics?

## How to Use

To reproduce the analysis, follow these steps:

1. Clone repository to local device.
2. Install Python and the required dependencies
3. Ensure dataset (e.g., `PCH_Palliative_Care_HOSPITAL.csv` in the directory.
5. Open then run Jupyter Notebook to execute analysis.

## Dependencies

The following Python libraries are used for this analysis:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

You can install these libraries using `pip`:

```
pip install -r requirements.txt
```

## Key Findings

1. Hospital Comparisons
- Dana-Farber Cancer Institute (MA) had the highest late chemotherapy rate (7%) but the lowest ICU admission rate (16%)
- Roswell Park Cancer Institute (NY) had the highest no-hospice rate (50%)
- University of Miami (FL) had the highest ICU admission rate (50%)
- USC Norris and City of Hope (CA) showed high ICU rates (>40%)

2. Geographic Patterns
- Massachusetts: Highest late chemo, lowest ICU rates
- New York: Highest no-hospice rates (48%)
- California, Texas, Florida: Highest ICU rates (>45%)
- Florida: Highest short hospice rate (30%)

3. Metric Relationships
- Strong positive correlation (0.79) between ICU admission and short hospice stays
- Moderate negative correlation (-0.32) between no-hospice use and short hospice stays
- Late chemotherapy showed weaker correlations with other metrics

4. Predictive Modeling
- ICU admission was the strongest predictor of short hospice stays
- Model achieved R² of 0.67, indicating moderately strong predictive power
- Simulation showed reducing ICU admission rates by 10% could improve short hospice stay rates by 3%
- Matching all centers to Dana-Farber's ICU rate could improve short hospice rates by 5.41%

5. Overall Performance Ranking
- Top performers: Roswell Park and Fred Hutchinson cancer centers
- Lowest performers: USC Kenneth Norris and City of Hope

6. Recommendations
- Emphasize earlier palliative care referrals
- Review ICU admission criteria and timing of hospice referrals
- Study high-performing centers to identify best practices
- Standardize palliative care approaches nationally to address regional variations



## License

The code and analysis in this repository are provided under the MIT License. Feel free to use, modify, and distribute the code as per the terms of the license.

## References/Resources
Palliative Care Datast: https://data.cms.gov/provider-data/dataset/qoeg-w7ck
Seaborn tutorial: https://seaborn.pydata.org/tutorial.html
Matplotlib tutorial: https://matplotlib.org/
Scikit tutorial: https://scikit-learn.org/stable/getting_started.html


**Note**: The data used in this analysis is for demonstration purposes and may not represent actual survey data.
