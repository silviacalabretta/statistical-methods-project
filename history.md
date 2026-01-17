# Data Cleaning

- I added a dictionary for the cities name, I think at the end we should delete these columns but now we have it,
- I deleted HashPassportNumber_p, UserID, HashEmail, BillID, BuyerMobile, TicketID columns
- about cancelDate I think creating new column is useless and it can cause overfitting as we have about 84% missing value that means they didn't cancel
- my suggestion is to create new column base on departureDate and createDate is more helpful to train model base on that
- for nationalcode I will add userCancelRate and the I will remove nationalCode 
- missing value in VehicleType is about 7% so I will drop those rows and maybe at the end we should delete this column as it is in Persian 
- about VehicleClass I will add unknown as a missing value


- I fount two tickets, one with price 0 and another with negative price (-1514000.0) so I deleted those rows, as all the other tickets have price from 40000 rials above
- For consistency I added a check on whether there was any ticket with departure time previous the created one, which would give a negative LeadTime, but luckily there were none, if you think of any other domain we have to check on other variables we can add it
- I also changed all the binary values to integers, so that plots and models work better (some were string, some were bool, now it's coherent)
- Instead of CouponDiscount I made a variable with the percentage of the discount, which is more general for the models
- From the DepartureTime which was deleted I rescued the moth of departure and the hour of departure, because I plotted the values and there were much more cancellations around the month of september and during the night hours, so they may be informative for the model (and then i deleted DepartureTime as it was alredy done
- In /src/data.py I wrote a function that splits the dataset in training and test set, and manages the creation of the user history: if a user only appear in the test set, we set User_Total_Tickets=0 and User_Cancel_Rate={average cancel rate in train set} (like a Bayesian prior).