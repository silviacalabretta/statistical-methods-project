# M - ‘Travel_ticket_cancellation’ Dataset

- Source: https://www.kaggle.com/datasets/pkdarabi/classification-of-travel-purpose

- Description: Every cancellation results in a fine for the ticket registration website by the airline. It is crucial to identify tickets likely to be canceled to manage cancellation risk effectively.

- Objective: Develop a model to predict if users will cancel their tickets. The response variable is Cancel (0 if not canceled, 1 if canceled).

Dataset variables:
Created: The timestamp indicates the ticket registration time.
CancelTime: The timestamp when the passenger canceled the ticket, if applicable.
DepartureTime: The scheduled departure time for the trip.
BillID: The unique identifier for the purchase transaction.
TicketID: The unique identifier for the ticket.
ReserveStatus: The payment status of the customer.
UserID: The unique identifier for the user.
Male: Indicates whether the ticket belongs to a male passenger or not.
Price: The ticket price without any discounts.
CouponDiscount: The discount applied by the passenger on the ticket.
From: The origin of the trip.
To: The destination of the trip.
Domestic: Indicates whether the trip is domestic or international.
VehicleType: Specifies details about the mode of transportation.
VehicleClass: Indicates whether the vehicle is first class or not.
Vehicle: Specifies the type of vehicle.
Cancel: Indicates whether the ticket has been canceled or not.
HashPassportNumber_p: Hashed version of the passport number.
HashEmail: Hashed version of the email address.
BuyerMobile : Hashed version of the buyer's mobile number.
NationalCode : Hashed version of the national identification number.
TripReason : The reason for the trip (1 = Work, 0 = Int).


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

- oversampling

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

