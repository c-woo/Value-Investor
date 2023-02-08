# Value Investor
 
I'll be using a LSTM model in order to predict the daily, weekly, and monthly prices of various companies in order to give investment recommendations for the year 2021. I am hoping to be able to make investment recommendations that only turn a profit for each company to maximize the performance of a portfolio that holds these stocks. The investment recommendations and performances of each company is in their respective notebooks.

I used PyTorch to implement the LSTM model and there is a TensorFlow application of the LSTM model in the PAMP notebook. All the necessary modules and functions are located in the data_functions file.

## Data
The data is from a set of portfolio companies trading data from emerging markets including 2020 Q1-Q4 and 2021 Q1 stock prices. Each company stock is provided in different sheets and each market's operating days varies based on the country of the company and the market the stocks are exchanged.
> **Date**  
> **Price:** Closing price of stock  
> **Open:** Opening price of stock  
> **High:** Highest price stock reached during day  
> **Low:** Lowest price stock reached during day  
> **Vol.:** Number of shares traded during day  
> **Change %:** Percentage change of closing price from previous day's close  

## BEEF
<img src='https://i.imgur.com/ME0yA73.jpg'>

- Daily: +13.5%
- Weekly: -3.5%
- Monthly: +2.1%

## CCB
<img src='https://i.imgur.com/2LS2O8W.jpg'>

- Daily: N/A
- Weekly: N/A
- Monthly: N/A

## DSMC
<img src='https://i.imgur.com/f0nxPAm.jpg'>

- Daily: +67.9%
- Weekly: +30.4%
- Monthly: +33.4%

## IMPJ
<img src='https://i.imgur.com/zAlWU7H.jpg'>

- Daily: +31.7%
- Weekly: +35%
- Monthly: +34.6%

## KCHOL
<img src='https://i.imgur.com/F4wyaGZ.jpg'>

- Daily: +16.3%
- Weekly: -0.5%
- Monthly: +5%

## MNHD
<img src='https://i.imgur.com/IrhOzwT.jpg'>

- Daily: +6.5%
- Weekly: -6.6%
- Monthly: +0.5%

## PAMP
<img src='https://i.imgur.com/gFWF8S1.jpg'>

- Daily: +25%
- Weekly: +21%
- Monthly: +17.5%

## SBER
<img src='https://i.imgur.com/NWiL0pt.jpg'>

- Daily: +10.8%
- Weekly: N/A
- Monthly: +8.5%

# Conclusion

I was able to create an investment strategy that resulted in a profit for almost all of the companies. The only company that I was unable to was CCB which had been on a downtrend for the whole first quarter of 2021. The daily predictions were usually the best performing for most of the companies except for IMPJ where the weekly predictions performed best. I would recommend using the daily predictions in order to maximize the performance of a portfolio that carries these stocks.








XaeqOE1iKHfhagtM