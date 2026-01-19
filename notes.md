Be careful about:
- we dropped the missing values of VehicleType, if we don't use it in the mdoels we can add them back in


Remember:
- 'Price': there was one ticket with price 0 and one with negative price (-1514000.0 rials), we deleted those rows. All the others have price equal or higher than 40000 rials.


Reasoning about data:
- a lot of tickets with moth of departure in august have a lot of days of lead time, which are summer vacation bought previously in spring.
- septemeber have a lot of cancellations because there were protests, so people wouldn't move / couldn't move/ didn't want to go to certain cities ...

Number of tickets per moth:
Month 1: 2
Month 2: 3264
Month 3: 6341
Month 4: 6264
Month 5: 8422
Month 6: 7522
Month 7: 10620
Month 8: 14029
Month 9: 15495
Month 10: 18329
Month 11: 3173
Month 12: 1

Most are in July-October: up to September it's summer, in October there were a lot of religious holidays. Also most of them were domestic, which makes sense:
Month 1: 2
Month 2: 3261
Month 3: 6320
Month 4: 6249
Month 5: 8401
Month 6: 7505
Month 7: 10566
Month 8: 13973
Month 9: 15426
Month 10: 18298
Month 11: 3166
Month 12: 1


KILL month of departure
Rename Price as LogPrice

Varibles to drop:
- Male: drop it
- User_Total_ticket: drop it
- Has_Discount: drop it because data is too imbalanced and there is only a 0.02% of difference between the two (created from CouponDiscount)
- VehicleClass: drop it because there is not much difference between class 0 and class 1, and also missing values (trains) add correlations, which is hell, so it's better to drop this variable
  
Okay variables:
- Tripreason: okay
- HourDeparture: build models with and without and test if it makes a difference
- Domestic: is imbalanced, so oversampling has to take care
- Vehicle: merge Plane and IntPlane (merge in Vehicle, still save in another column the actual data)
-LeadTime_Days: okay
- From_Rate and To_Rate OR Route_Rate

Check correlation between not domestic tickets with international plane, and in case merge plane and IntPlane to remove correlation

check User_Cancel_Rate on people with multiple tickets




TO DO:
- delete randomly some data to decrease the cancellation rate in the month of september
- change the splitting function, so that it is random
- merge Vehicle_Plane and Vehicle_InternationalPlane
Bus: 48960
Train: 38441
Plane: 12809
IntPlane: 795
