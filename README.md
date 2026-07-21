# Retail Demand Forecasting





## Strategy

I used a a bottom-up forecasting approach where I modeled daily demand and then aggregated the predictions weekly. Upon research, I found that most online retailers keep daily models but change how they evaluate and use it. This strategy is best for businesses needing both granular daily insight and accurate weekly planning as it retains daily trend shapes. 

**What I would do next**

- Route the data to the right teams: Feed the daily predictions to logistics teams who need to know if Monday is busier than Friday. Feed the aggregated 25.57% weekly predictions to procurement teams for inventory ordering.

- Optimize the daily model using a weekly loss function: Instead of forcing the model to get every Tuesday perfect, I would tweak the model's hyperparameters to minimize the weekly aggregated error.


## Set Up for MacOS
```bash
# clone repo:
git clone https://github.com/machaniG/retail-demand-forecasting.git

# create virtual environement
python -m venv venv
source venv/bin/activate

# install requirements
pip install -r requirements.txt
```