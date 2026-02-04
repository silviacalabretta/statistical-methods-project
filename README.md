# Statistical Methods Project: Travel Ticket Cancellation Prediction

## Project Overview

This project focuses on the **'Travel_ticket_cancellation'** dataset. The primary objective is to develop a statistical model to predict whether a passenger will cancel their ticket. 

Identifying potential cancellations is crucial for risk management, as every cancellation results in a fine for the ticket registration website from the airline.


## Dataset

* **Source:** [Kaggle - Classification of Travel Purpose](https://www.kaggle.com/datasets/pkdarabi/classification-of-travel-purpose)

* **Goal:** Binary Classification (Target: `Cancel` — 0 if not canceled, 1 if canceled).

* **Description:** Every cancellation results in a fine for the ticket registration website by the airline. It is crucial to identify tickets likely to be canceled to manage cancellation risk effectively.

* **Objective:** Develop a model to predict if users will cancel their tickets. The response variable is Cancel (0 if not canceled, 1 if canceled).

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
- VehicleClass: Indicates whether the vehicle is first class or not (False: 0, True: 1, 'Unknown': 2).
- Vehicle: Specifies the type of vehicle ('Bus': 0, 'Train': 1, 'Plane': 2, 'InternationalPlane': 3).
- Cancel: Indicates whether the ticket has been canceled or not.
- HashPassportNumber_p: Hashed version of the passport number.
- HashEmail: Hashed version of the email address.
- BuyerMobile : Hashed version of the buyer's mobile number.
- NationalCode : Hashed version of the national identification number.
- TripReason : The reason for the trip (1 = Work, 0 = Int).


## Repository Structure

The project is organized into the following directories to separate data, code, and results:

```text
statistical-methods-project/
├── data/                # Dataset files (raw and processed)
├── notebooks/           # Jupyter Notebooks for EDA and modeling
├── plots/               # Generated figures (barplots, distributions, etc.)
├── src/                 # Source code for feature selection and helper functions
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation

```

## Project Development 

### 1. Data exploration & feature engineering

We performed extensive Exploratory Data Analysis (EDA) to understand the variables and prepare the data for modeling.

**Variable analysis.** Enforced domain constraints on variables and examined distributions.

**Effectless variables.** Remove the following variables:
- `HashPassportNumber_p`, `UserID`, `HashEmail`, `BillID`, `BuyerMobile`, `TicketID`: identification variables, not useful to the prediction of the cancellation probability;
- `CancelTime`: contains data only about the cancelled tickets, if we would add this variable to the model we wold have target leakage.
- `VehicleType`: there are too many categories in this variable, it's not possible to retrieve any information about cancellation.

**Target leakage.**
- `CancelTime`: contains data only about the cancelled tickets.
- `ReserveStatus`: we don't have information about the categories meaning and some categories have high cancellation rates, while others have 0%.

**Correlation check.** 
* `InternationalPlane` and `Domestic`=False: we merged the categories `InternationalPlane` and `Plane` under the `Vehicle` feature.
- `Vehicle`=Train and `VehicleClass` missing: we removed `VehicleClass` because not much informative.

**Feature engineering:**
- `LeadTime_Days` by calculating the difference between `DepartureTime` and `CancelTime`
- `TimeOfDay`: departure hour of the ticket, binned in Morning, Afternoon, Evening and Night.
- `Route`: traveling path, combination of `From` and `To`.
- `PercentageDiscount`.

**Target encoding.** Created the new values `UserRate`, `From_Rate`, `To_Rate`, `Route_Rate` value, using Leave-One-Out Smoothed Rate on training set and classic Smoothed Rate on test set.


#### Final dataset variables

| Variable | Description |
| --- | --- |
| `Cancel` | **Target Variable** (0 = No, 1 = Yes) |
| `LogPrice` | Log of ticket price (excluding discount) |
| `LogLeadTime` | Log of advance days the ticket was booked |
| `TimeOfDay` | Hour of departure binned in Morning, Afternoon, Evening,Night |
| `TripReason` | Reason for trip (Work vs Int) |
| `Domestic` | Whether the trip is domestic or international |
| `Vehicle` | Travle vehicle, Bus, Train or Plane |
| `From_Encoded` | The origin of the trip |
| `To_Encoded` | The destination of the trip |
| `Route_Encoded` | The travel path of the trip |


### 2. Handling imbalanced data

Since cancellations are the minority class ($15.2\%$), we implemented strategies to handle class imbalance:

* **Class weights:** Applied `class_weight='balanced'` in models (e.g., Logistic Regression) to penalize misclassifying the minority class.
* **Resampling techniques:** Considered undersampling and SMOTE (oversampling) to synthesize minority samples.
* **Evaluation metrics:** Shifted focus from simple Accuracy to **F1-Score**, **Precision/Recall**, and **ROC-AUC** to better evaluate performance on the minority class.

### 3. Statistical modeling

We explored various statistical and machine learning models to identify the best predictor:

* **GLM:** Generalized Linear Models and Generalized Additive Models for non-linear relationships.
* **Regularized regression:** Logistic Regression with L1/L2 regularization.
* **Tree-based models:** Random Forests.

### 4. Model evaluation

Models were compared using:

* **ROC Curve & AUC:** To measure separation capability.
* **Confusion matrix:** To visualize False Positives vs. False Negatives (prioritizing recall to capture cancellations).

## Installation and Usage

To replicate this analysis on your local machine:

1. **Clone the repository:**
```bash
git clone [https://github.com/silviacalabretta/statistical-methods-project.git](https://github.com/silviacalabretta/statistical-methods-project.git)
cd statistical-methods-project

```


2. **Install dependencies:**
Ensure you have Python installed, then run:
```bash
pip install -r requirements.txt

```


3. **Run the Notebooks:**
Launch Jupyter Lab or Notebook to view the analysis:
```bash
jupyter notebook notebooks/

```
