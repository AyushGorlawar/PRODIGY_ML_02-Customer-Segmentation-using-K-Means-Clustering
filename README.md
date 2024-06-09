# PRODIGY_ML_02

# Customer Segmentation using K-Means Clustering

This project demonstrates the use of K-Means clustering to group customers of a retail store based on their purchase history.

## Dataset

The dataset used for this project is from Kaggle and can be found [here](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python).

## Files

- `customer_segmentation.py`: Python script for performing K-Means clustering and visualizing the results.
- `Mall_Customers.csv`: Dataset file containing customer data.

## Requirements

- Python 3.x
- pandas
- matplotlib
- seaborn
- scikit-learn

## Installation

Install the required Python libraries using pip:

```bash
pip install pandas matplotlib seaborn scikit-learn
```
## Usage
- Download the dataset and save it as Mall_Customers.csv.
- Run the customer_segmentation.py script:
``` bash
python customer_segmentation.py
```
## Steps in the Script
- Load and explore the dataset.
- Preprocess the data by dropping irrelevant columns and normalizing the data.
- Use the Elbow method to determine the optimal number of clusters.
- Apply K-Means clustering to segment customers.
- Visualize the resulting clusters.

## Results
The script generates a scatter plot showing customer segments based on annual income and spending score.

## License
This project is licensed under the [MIT Licensehere](https://github.com/AyushGorlawar/PRODIGY_ML_02/blob/main/LICENSE)
