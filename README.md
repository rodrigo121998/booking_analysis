# Booking Data Analysis

This project involves analyzing a booking dataset to understand and predict booking cancellations. The analysis includes data exploration, feature engineering, data processing, modeling, and evaluation.

## Dataset

The dataset contains information about bookings with the following features:

- **id**: Unique identifier for each booking.
- **lead_time**: Time between booking date and reservation date (in days).
- **arrival_week**: Week number of the arrival date.
- **duration**: Booking duration (in days).
- **prev_cancel**: Number of previous bookings that were cancelled by the customer prior to the current booking.
- **booking_changes**: Number of changes/amendments made to the booking between booking date and reservation/cancellation date.
- **waiting_period**: Waiting period for booking confirmation (in days).
- **per_day_price**: Per night booking price (in US $).
- **parking**: Number of car parking spaces required by the customer.
- **special_request**: Number of special requests made by the customer.
- **segment**: Market segment designation.
- **deposit**: Whether the customer made a deposit to guarantee the booking.
- **cust_type**: Type of booking.
- **is_cancelled**: Value indicating if the booking was cancelled (1) or not (0).

## Data Processing and Feature Engineering

1. **Loading the Data**: The dataset is loaded into a pandas DataFrame.
2. **Feature Engineering**: 
    - A new feature `total_cost` is created by multiplying `duration` and `per_day_price`.
3. **One-Hot Encoding**: Categorical features (`segment`, `deposit`, `cust_type`) are encoded using one-hot encoding.

## Modeling

1. **Feature Selection**: 
    - Selected features based on correlation analysis: `lead_time`, `booking_changes`, `prev_cancel`, `waiting_period`, `total_cost`, `special_request`, `segment_Offline TA/TO`, `segment_Online TA`, `deposit_Non Refund`, `deposit_Refundable`, `cust_type_Group`, `cust_type_Transient`, `cust_type_Transient-Party`.
2. **Data Splitting**: 
    - The data is split into training and testing sets.
3. **Model Training**: 
    - A RandomForest Classifier is trained on the training data.
4. **Model Evaluation**: 
    - Model performance is evaluated using accuracy, confusion matrix, classification report, and ROC AUC score.
5. **Feature Importance**: 
    - The importance of features is analyzed to understand their impact on the predictions.

## Results

- **Accuracy**: Indicates the proportion of correct predictions.
- **Confusion Matrix**: Provides insight into true positives, true negatives, false positives, and false negatives.
- **Classification Report**: Offers precision, recall, and F1-score for each class.
- **ROC AUC**: The ROC curve and AUC score provide a graphical representation of the model's performance.

## Conclusion

The analysis provides a comprehensive understanding of booking cancellations. The RandomForest Classifier performed well, with a good balance of precision and recall. The feature importance analysis highlighted the key drivers of booking cancellations.
