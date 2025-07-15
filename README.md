# Laboratory #9 - Missing Data and Feature Engineering

## Description
This project addresses missing data handling and feature engineering techniques using the Titanic dataset. The analysis includes:

1. Missing data report
2. Imputation methods with justification
3. Normalization technique comparison
4. Result visualization

## Data
The project uses two datasets:
- `titanic_MD.csv`: Dataset with missing values for imputation practice
- `titanic.csv`: Original dataset for comparison

## Requirements
- R (version 4.0 or higher)
- The following R packages:
  - `caret`
  - `dplyr`

## Installation
1. Clone this repository or download the files
2. Ensure you have both `titanic_MD.csv` and `titanic.csv` in your working directory
3. Install required packages if needed:
```r
install.packages(c("caret", "dplyr"))
```

## Usage
Run the R Markdown file (`Laboratorio9.Rmd`) to:
1. Generate missing data reports
2. Apply imputation methods
3. Compare normalization techniques
4. Visualize results

## Implemented Methods
### Missing Data Handling:
- Median imputation (for continuous variables like Age)
- Mode imputation (for discrete variables like SibSp and Parch)
- Outlier handling using percentiles

### Normalization:
- Z-Score standardization
- Min-Max scaling
- MaxAbs scaling

## Results
The analysis includes:
- RMSE reports for imputation method comparison
- Normalization technique comparisons
- Before/after processing distribution visualizations

## Author
Christian Barrios

## Date
2024-11-15
