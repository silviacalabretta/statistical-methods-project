# M - ‘Travel_ticket_cancellation’ Dataset

- Source: https://www.kaggle.com/datasets/pkdarabi/classification-of-travel-purpose

- Description: Every cancellation results in a fine for the ticket registration website by the airline. It is crucial to identify tickets likely to be canceled to manage cancellation risk effectively.

- Objective: Develop a model to predict if users will cancel their tickets. The response variable is Cancel (0 if not canceled, 1 if canceled).

Dataset variables:

- Created: The timestamp indicates the ticket registration time.
- CancelTime: The timestamp when the passenger canceled the ticket, if applicable.
- DepartureTime: The scheduled departure time for the trip.
- BillID: The unique identifier for the purchase transaction.
- TicketID: The unique identifier for the ticket.
- ReserveStatus: The payment status of the customer.
- UserID: The unique identifier for the user.
- Male: Indicates whether the ticket belongs to a male passenger or not.
- Price: The ticket price without any discounts.
- CouponDiscount: The discount applied by the passenger on the ticket.
- From: The origin of the trip.
- To: The destination of the trip.
- Domestic: Indicates whether the trip is domestic or international.
- VehicleType: Specifies details about the mode of transportation.
- VehicleClass: Indicates whether the vehicle is first class or not.
- Vehicle: Specifies the type of vehicle.
- Cancel: Indicates whether the ticket has been canceled or not.
- HashPassportNumber_p: Hashed version of the passport number.
- HashEmail: Hashed version of the email address.
- BuyerMobile : Hashed version of the buyer's mobile number.
- NationalCode : Hashed version of the national identification number.
- TripReason : The reason for the trip (1 = Work, 0 = Int).


# Project structure

## Data exploration
Feature engineering:
- check if cancel column is congruent with missing cancel time
- create another variable: set cancel time = departure datetime-cancel datetime and in case divide it in intervals based on the range
- TripReason: turn it in 0,1 (it is Work or Int)
- do all trains ha ìve null VehicleClass? 

- hypothesized important variables:

- useless variables:
  - BillID
  - HashPassportNumber_p
  - HashEmail

### Oversampling
1. Class Weights: Instead of changing the data, you change the math. You tell the model: "Making a mistake on a Cancellation (Class 1) is 10x worse than making a mistake on a non-cancellation." Most scikit-learn models (Logistic Regression, Random Forest, SVM) have a built-in parameter for this.
\[
`model = LogisticRegression(class_weight='balanced')`
\]
2. Resampling (changing the data)
- Undersampling: You randomly delete rows from the majority class (Not Canceled) until it matches the minority class.

Pros: Fast training.

Cons: You throw away valuable data (bad for small datasets).

- Oversampling (SMOTE): You synthesize artificial new examples of the minority class.

Pros: Keeps all data.

Cons: Can introduce noise and overfitting.

3. Change the Metric
Never use "Accuracy" for imbalanced data. If 95% of users don't cancel, a dummy model has 95% accuracy.

Use these instead:

- F1-Score: The harmonic mean of Precision and Recall.

- ROC-AUC: Measures how well the model separates the two classes.

- Precision-Recall AUC: Often better than ROC for highly imbalanced datasets.


**In our case:**

Start with class_weight='balanced'. It is the simplest, requires no extra libraries, and doesn't destroy or fake any data. It usually gives a massive boost in detecting the minority class immediately.

Combine it with the right metric. Focus on maximizing the F1-Score or Recall (if you care more about catching all cancellations, even if you flag some false alarms).

## Models
- GLM GAM non-linearities, regularized logistic regression, probit regression, trees, ...
- significance test, p-values, evaluation

Possible povs: predictions based on the line, predictions based on the people.

Notice: False Positive is better than False Negative, we prefer to predict someone is goinc to cancel even if he won't rather than to no

## Comparison between models
- AUC curve
- ROC curve + AUC
- Precision / Recall
- Confusion matrix
  
## Possible exploration
- check if there is some user that is cancelling a lot, so that he is very prone to cancelling: use the NationalCode (not missing values)
- check whether there is a correlation between price and correlations

