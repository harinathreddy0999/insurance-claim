# Insurance Claim Prediction Model

This project demonstrates a machine learning pipeline to predict whether an individual will file an insurance claim or not based on various features like age, vehicle information, and previous insurance history. It uses a Random Forest Classifier to perform the prediction and is built using Python and various machine learning libraries.

## Libraries Used

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning and data preprocessing
- **matplotlib**: Data visualization
- **seaborn**: Statistical data visualization
- **joblib**: Model serialization
- **dotenv**: Managing environment variables

## Data Overview

The dataset consists of 382,154 records and 11 features, which include:

- `Claim ID`: Unique identifier for each claim
- `Gender`: Gender of the individual (Male/Female)
- `Age`: Age of the individual
- `Driving_License`: Whether the individual has a driving license (1 = Yes, 0 = No)
- `Region_Code`: Code representing the region
- `Previously_Insured`: Whether the individual was previously insured (1 = Yes, 0 = No)
- `Vehicle_Age`: Age of the vehicle (Categorical: <1 Year, 1-2 Year, >2 Years)
- `Previous_Vehicle_Damage`: Whether the vehicle has had previous damage (Yes/No)
- `Annual_Premium`: Annual premium paid by the individual
- `Policy_Sales_Channel`: Sales channel through which the policy was sold
- `Response`: The target variable (1 = Claim filed, 0 = No claim)

## Setup

1. Clone the repository:

```bash
git clone https://github.com/your-username/insurance-claim-prediction.git
```

2. Install the required libraries:

```bash
pip install -r requirements.txt
```

3. Prepare your dataset in the appropriate path as defined in the code.

4. Run the code by executing:

```bash
python insurance_claim_model.py
```

This will train the model, evaluate its performance, and save the trained model and scaler as `.pkl` files.

## Steps in the Pipeline

1. **Data Loading**: The dataset is loaded using pandas and basic exploration is done to check for missing values and column types.
   
2. **Data Preprocessing**:
   - Label encoding is used for categorical variables like `Gender`, `Vehicle_Age`, and `Previous_Vehicle_Damage`.
   - Missing data is handled by dropping null values.
   - Feature scaling is done using `StandardScaler` to standardize the dataset.

3. **Model Training**:
   - A Random Forest Classifier is used to predict the target variable (`Response`).
   - The data is split into training and testing sets (80% train, 20% test).

4. **Evaluation**:
   - The model's performance is evaluated using accuracy, classification report, and confusion matrix.
   - Feature importance is visualized using a bar plot.

5. **Model Saving**:
   - The trained model and the scaler are saved as `.pkl` files using `joblib`.

## Evaluation Metrics

The model achieves the following evaluation results:

- **Accuracy**: 89%
- **Precision**: 77% (for the positive class)
- **Recall**: 45% (for the positive class)
- **F1-score**: 57% (for the positive class)

## Conclusion

The model performs reasonably well in predicting insurance claims, with a strong precision for non-claimants. Further improvements can be made by fine-tuning the model and trying different algorithms to increase recall and F1-score for the positive class.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- scikit-learn for providing the machine learning tools
- Pandas and NumPy for data manipulation
