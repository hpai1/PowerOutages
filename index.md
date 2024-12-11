# Analyzing Power Outages Across the United States

### Hrithik Pai and Gautham Kishore

## Introduction

For this project, we analyzed a dataset of power outages in the United States between January 2000 and July 2016.

The dataset was provided by Purdue University and has various information about power outages, allowing for in-depth analysis.

We started by cleaning and tidying the data, which included handling formatting issues and gaining an overall idea of the data it contains.

Then we analyzed the missingness of the data and handled missing values accordingly. After that, we built a regression model that predicted the number of customers that were affected by a power outage.

The original dataframe had 1534 rows and 56 columns, but we only kept the ones relevant to our model and problem. A couple of them are shown below.

| Column Name        | Description                                                                                   |
| ------------------ | --------------------------------------------------------------------------------------------- |
| U.S.\_STATE        | The U.S. state where the power outage occurred.                                               |
| NERC.REGION        | The North American Electric Reliability Corporation (NERC) region associated with the outage. |
| CLIMATE.REGION     | The climate region where the power outage occurred.                                           |
| ANOMALY.LEVEL      | The anomaly level associated with the outage (e.g., weather-related or system failures).      |
| CAUSE.CATEGORY     | The category of the cause for the power outage (e.g., weather, equipment failure).            |
| OUTAGE.DURATION    | The total duration of the power outage in hours.                                              |
| DEMAND.LOSS.MW     | The amount of power demand lost during the outage, measured in megawatts (MW).                |
| CUSTOMERS.AFFECTED | The number of customers affected by the power outage.                                         |
| TOTAL.PRICE        | The total price of electricity during the outage period.                                      |
| TOTAL.SALES        | The total electricity sales during the outage period.                                         |
| POPPCT_URBAN       | The percentage of the population living in urban areas in the affected region.                |
| POPDEN_URBAN       | The population density of urban areas in the affected region (persons per square mile).       |
| AREAPCT_URBAN      | The percentage of the area classified as urban in the affected region.                        |

## Data Cleaning and Exploratory Data Analysis

### Data Cleaning Process

Our initial dataset contained 1,534 power outage records with 20 columns. The cleaning process focused on three key areas: timestamp standardization, handling of missing/zero values, and feature engineering.

First, we standardized the temporal data by combining separate date and time columns (`OUTAGE.START.DATE`, `OUTAGE.START.TIME`, `OUTAGE.RESTORATION.DATE`, `OUTAGE.RESTORATION.TIME`) into unified datetime columns. The original data contained 9 null values in start times and 58 null values in restoration times, which were preserved as `NaT` (Not a Time) in the combined columns for transparency.

Next, we addressed data quality by converting zeros to `NaN` in critical measurement columns: outage duration, customers affected, and demand loss. This increased our missing value counts but provided a more accurate representation of the data, as true zero values in these fields were likely recording errors - a power outage would typically affect at least some customers and last for some duration.

Finally, we performed feature engineering by creating a composite urban metric. This combined three correlated urban indicators (urban population percentage, urban population density, and urban area percentage) into a single standardized score. The resulting URBAN column provides a more concise representation of an area's urbanization level, with higher values indicating more urbanized areas. The feature has 50 unique values corresponding to different states' urbanization levels.

After cleaning, our dataset maintained its 1,534 records but with a more focused set of 16 columns. Key missing value counts in the final dataset include:

- CUSTOMERS.AFFECTED: 655 missing values
- DEMAND.LOSS.MW: 901 missing values
- OUTAGE.DURATION: 136 missing values

<iframe src="head_md.md" width="800" height="600" frameborder="0"></iframe>

## Exploratory Data Analysis

### Distribution of Outage Durations

<iframe src="distribution_duration.html" width="800" height="600" frameborder="0"></iframe>

The histogram of outage durations reveals a highly right-skewed distribution, with the majority of outages being resolved within 20,000 minutes (approximately 2 weeks). However, there are notable outliers extending up to 100,000 minutes (about 69 days), suggesting that while most outages are resolved relatively quickly, some extreme cases require extended periods for restoration.

### Causes of Power Outages

 <iframe src="distribution_causes.html" width="800" height="600" frameborder="0"></iframe>

The bar chart of outage causes shows that **severe weather** is by far the most common cause, accounting for nearly half of all recorded outages. This is followed by **intentional attacks** as the second most frequent cause, with other categories like system operability disruption, public appeal, and equipment failure occurring less frequently. Islanding and fuel supply emergencies are the least common causes in our dataset.

### Duration and Impact Analysis

<iframe src="duration_impact.html" width="800" height="600" frameborder="0"></iframe>

The scatter plot of customers affected versus outage duration reveals several interesting patterns:

- Most outages cluster in the lower left corner, indicating that typical outages affect fewer than 500,000 customers and last less than 20,000 minutes
- Severe weather events (shown in blue) dominate the upper ranges of both duration and customer impact
- System operability disruptions (green) show high variation in customer impact but tend to have shorter durations
- Intentional attacks (red) generally affect fewer customers and have shorter durations compared to weather-related events

### Regional Patterns

<iframe src="regional_patterns.html" width="800" height="600" frameborder="0"></iframe>

The box plot of outage durations across climate regions shows substantial variation:

- All regions show significant outliers, particularly visible in the East North Central and West regions
- The West North Central region shows the most compact distribution, suggesting more consistent restoration times
- The Northeast and Southeast regions show similar median durations but different outlier patterns
- The Central region shows a higher median duration compared to other regions

### Regional Trends Over Time

<iframe src="pivot_table.md" width="800" height="400" frameborder="0"></iframe>

Looking at the relationship between outage duration and climate regions over time (shown in our pivot table), we observe varying patterns of restoration efficiency across different regions. The East North Central region, for example, shows a trend toward shorter outage durations in recent years, while the West region exhibits more variable restoration times. This regional variation could be attributed to differences in infrastructure resilience, local response capabilities, and the predominant types of outage causes in each region.

### Summary Statistics by Cause Category

The aggregated statistics reveal that while severe weather events had the highest average customer impact (190,972 customers), fuel supply emergencies had the longest average duration (13,484 minutes) but affected very few customers. System operability disruptions showed a notably high customer impact (211,066 on average) despite relatively short durations (747 minutes), suggesting efficient resolution protocols for these types of incidents.

## Assessment of Missingness

In our power outage dataset, **DEMAND.LOSS.MW** (58.74% missing) exhibits strong characteristics of being **Not Missing at Random** (**NMAR**). We believe this missingness mechanism is **NMAR** because the likelihood of missing demand loss measurements could directly depend on the unobserved value itself - during catastrophic outages, measurement systems might fail precisely when the demand loss is extremely high, or conversely, very small power losses might go unrecorded due to being below measurement thresholds. To potentially convert this **NMAR** pattern to **MAR**, we would need additional data about each utility's measurement capabilities, including their minimum threshold for recording demand loss and whether their monitoring systems have backup power during outages. This technical metadata would likely explain much of the missingness pattern, making it **MAR** with respect to these new variables.

Our missingness analysis focused on exploring dependencies between the missingness of **ANOMALY.LEVEL** and other variables in the dataset, particularly **MONTH** and **CUSTOMERS.AFFECTED**. The permutation test results revealed a **significant dependency** between ANOMALY.LEVEL's missingness and MONTH (**p-value < 0.05**), indicating a **MAR** pattern. However, when testing against CUSTOMERS.AFFECTED, we found **no significant dependency** (**p-value = 0.3008**), despite an observed difference of -95,203.17 in means. This suggests that while there might be a practical relationship between customer impact and missing anomaly levels, it's not statistically significant enough to conclude MAR based on this variable alone.

<iframe src="missingness_distribution.html" width="100%" height="500" frameborder="0"></iframe>
<iframe src="missingness_permutation.html" width="100%" height="500" frameborder="0"></iframe>

The distribution plots from our analysis provide visual evidence of these relationships. Looking at the visualization above, we can see distinct differences in the distribution of **MONTH** values between cases where ANOMALY.LEVEL is missing versus not missing. The permutation test visualization shows the empirical distribution of test statistics, with our observed difference marked by the red line. This distribution helps us understand the significance of our findings - while the **MONTH** relationship shows clear separation between missing and non-missing cases, the **CUSTOMERS.AFFECTED** relationship exhibits more overlap, supporting our statistical conclusions about their respective dependencies.

## Statistical Analysis of Power Outage Patterns

### Hypothesis Test 1: Impact of Severe Weather on Outage Duration

For our first hypothesis test, we examined whether severe weather conditions lead to longer power outage durations compared to other causes. We formulated the following hypotheses:

- **Null Hypothesis (H₀)**: There is no difference in outage duration between severe weather and non-severe weather causes
- **Alternative Hypothesis (H₁)**: Severe weather causes lead to longer outage durations

We chose a permutation test with the difference in means as our test statistic, as it makes no assumptions about the underlying distribution of outage durations and is robust to outliers. Using 1,000 permutations and a significance level of α = 0.05, we obtained a p-value of 0.0000, with an observed difference in means of 2,399.86 minutes (approximately 40 hours). The extremely low p-value suggests strong evidence against the null hypothesis, indicating that outages caused by severe weather tend to last significantly longer than those caused by other factors. However, we should note that this does not prove causation, as there may be other confounding variables affecting outage duration.

### Hypothesis Test 2: Regional Variations in Customer Impact

Our second analysis investigated whether different climate regions experience varying levels of customer impact during power outages:

- **Null Hypothesis (H₀)**: There is no relationship between climate region and number of customers affected
- **Alternative Hypothesis (H₁)**: Different climate regions have different numbers of customers affected

We used a permutation test with the sum of squared deviations from the overall mean as our test statistic, which effectively captures the variation between regions. With 1,000 permutations and α = 0.05, we obtained a p-value of 0.0890. While this result doesn't meet the conventional threshold for statistical significance, it suggests a potential relationship between climate regions and the scale of customer impact. The South and West regions showed notably higher average numbers of affected customers, though we cannot conclusively state that climate region is the causal factor for these differences.

<iframe id="hypothesis-viz" src="hypothesis_test_viz.html" width="100%" height="600px" frameborder="0"></iframe>

## Framing the Prediction Model

To predict the number of **CUSTOMERS.AFFECTED** during an outage, we propose a regression model that estimates the impact based on key features available at the early stages of the event. Given the nature of the problem, we will use **Mean Absolute Error (MAE)** as the evaluation metric. MAE is particularly suitable for this context because it provides a straightforward interpretation of the average absolute difference between predicted and actual values, which is critical for assessing the practical implications of prediction errors.

---

### Key Considerations

#### 1. Feature Selection

The features included in the model are chosen based on their availability and relevance during the early stages of an outage. This ensures that the model predictions align with the information known _before restoration_. Features that become available only after the outage is resolved (e.g., restoration time or outage duration) are excluded to prevent data leakage and maintain real-time applicability.

#### 2. Target Variable

The target variable is **CUSTOMERS.AFFECTED**, which represents the number of individuals or entities impacted by the outage.

#### 3. Prediction Use Case

The prediction model will assist utility companies, policymakers, and emergency services in planning and allocating resources efficiently during an outage. Accurate early predictions can help mitigate the impact of outages by improving response times and resource allocation.

---

### Relevant Features

The following features are included in the model because they are both available during the early stages of an outage and have predictive relevance:

- **YEAR**: The year of the outage (e.g., 2020, 2021).
- **MONTH**: The month of the outage (e.g., January, February, etc.).
- **U.S.\_STATE**: The U.S. state where the outage occurred (e.g., California, Texas).
- **NERC.REGION**: The NERC region, which represents the power grid region (e.g., Western, Eastern).
- **CLIMATE.REGION**: The climate region of the affected area (e.g., Coastal, Desert).
- **CAUSE.CATEGORY**: The cause of the outage (e.g., weather, equipment failure).
- **is_climate_missing**: A binary indicator that marks whether climate data is missing for the outage.
- **is_missing**: A general missingness indicator that applies to the features used in the model.

---

### Evaluation Metric

We will use **Mean Absolute Error (MAE)** as the evaluation metric for the model. MAE is defined as:

$
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |\hat{y}_i - y_i|
$

where:

- $ \hat{y}\_i $: Predicted value
- $ y_i $: Actual value
- $ n $: Total number of predictions

This metric is particularly effective for our use case because it provides a clear, interpretable measure of the average prediction error in the same units as the target variable (**CUSTOMERS.AFFECTED**).

## Baseline Model

For our Baseline model, we predicted number of customers affected by power outages (`CUSTOMERS.AFFECTED`) using a limited set of features: the year of the outage (`YEAR`) and the cause category (`CAUSE.CATEGORY`).

### Dataset Preprocessing

- **Target Variable Handling**: Rows with missing values in the target variable (`CUSTOMERS.AFFECTED`) were removed to ensure the model has valid training data.
- **Feature Selection**:
- `YEAR` was included as a numerical feature to account for temporal trends in outages.
- `CAUSE.CATEGORY` was included as a categorical feature to capture the impact of different outage causes on the number of affected customers.

We decided to split the data into 80% training data and 20% testing data.

To make the code easier to follow and debug, we created pipelines for each of the above steps.

1. **Preprocessor**:

- **One-Hot Encoding for `CAUSE.CATEGORY`**: Converts the categorical feature into a binary representation to be compatible with the linear regression model.
- **Passthrough for `YEAR`**: Retains the numeric feature without any transformation.

2. **Regressor**:

- A **Linear Regression** model was chosen as the baseline due to its simplicity and interpretability. It establishes a direct linear relationship between the features and the target variable.

### Model Training

The pipeline was fitted on the training data, learning the relationships between the input features (`YEAR` and `CAUSE.CATEGORY`) and the target variable (`CUSTOMERS.AFFECTED`).

### Results

The calculated **Mean Absolute Error** for this baseline model was:
$
\text{MAE} = {132425.95}
$
This means that the difference between the predicted and actual number of affected customers was 132426 on average.

## Final Model

For our final model, we expanded on our baseline model to include additional features such as seasonal variations (`Season`), regional attributes (`U.S._STATE` and `CLIMATE.REGION`), and a flag indicating missing climate data (`is_climate_missing`). These enhancements were aimed at capturing more contextual and environmental factors affecting power outages.

We added a `Season` feature which was calculated from the Month column, allowing the model to account for seasonal trends. Also, we decided to include regional attributes such as `U.S._STATE` and `CLIMATE.REGION` for locational trends. We also added a `is_climate_missing` flag to handle and encode missing information in the `CLIMATE.REGION` feature.

For the actual model itself, we imputed numerical features with the mean and were standardized. For categorical features, we use One-Hot Encoding so that the Random Forest model would be more accurate.

We also used Grid Search to test various hyperparameters such as `n_estimators`, `max_depth`, and `min_samples_split`.

### Results

The calculated **Mean Absolute Error** for this final model was:
$
\text{MAE} = {674.76}
$
where the Best Hyperparameters were `regressor__min_samples_split`: 2 and `regressor__n_estimators`: 200.

We can definitely see that this model was a significant improvement on the previous one because the MSE went down by a lot.

# Fairness Analysis: Rural vs. Urban States

In our fairness analysis, we examined whether our power outage prediction model exhibited any systematic differences in performance between rural and urban states. States were classified as **urban** if they had a population density generally exceeding **200 people per square mile** (including states like New Jersey, Massachusetts, and California), while states with lower population densities were classified as **rural**. Our test utilized **Mean Absolute Error (MAE)** as the primary metric, which represents the average magnitude of prediction errors, with lower values indicating better model performance. We employed a **permutation test** with **100,000 permutations** to ensure high precision in our statistical assessment.

<iframe src="fairness_performance_comparison.html" width="800" height="600" frameborder="0"></iframe>

The results revealed an interesting pattern in our model's predictive performance. For rural states, the model achieved an **MAE of 202.92**, while for urban states, the MAE was substantially higher at **416.83**, resulting in an observed difference of **-213.90** (rural MAE minus urban MAE). This indicates that our model's predictions were, on average, more accurate for rural states than for urban states, with predictions deviating by about 214 fewer customers on average in rural states. The test included a robust sample size for both groups, with **118 samples** from rural states and **186 samples** from urban states, providing a reliable basis for comparison.

With our highly precise permutation test using 100,000 permutations, we obtained a **p-value of 0.0568**, which is slightly above the conventional significance threshold of 0.05. This means we **fail to reject** the null hypothesis that the model's performance is the same for rural and urban states. This result, with its high precision due to the large number of permutations, suggests that while there appears to be a substantial practical difference in the model's performance between rural and urban states, we cannot conclusively say this difference is systematic rather than due to chance. The fact that the p-value is very close to but not quite at the significance threshold indicates that this difference in performance warrants careful consideration in real-world applications, even if it doesn't meet the strict criteria for statistical significance.
