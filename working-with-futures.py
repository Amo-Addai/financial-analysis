import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from quantopian.research.experimental import history, continuous_future


"""
SAMPLE COMMODODITIES, INDEX, CURRENCY (FOREX) FUTURES ARE ALL AVAILABLE ON QUANTOPIAN
"""

future_contract = symbols('NGF20')  # NG (Natural Gas) F (January Future) 20 (2020)
for key in future_contract.to_dict():
    print("{} -> {}".format(key, future_contract.to_dict()[key]))

futures_position_value = get_pricing(future_contract, start_date='2017-01-01', end_date='2018-01-01')
futures_position_value.name = futures_position_value.name.symbol
futures_position_value.plot()
# NOW, LET'S GET SOME HISTORICAL DATA
ngf20 = future_contract  # RESET A NEW SHORTER VARIABLE :')
ngf20_data = history(ngf20, fields=['price', 'open_price', 'high', 'low', 'close_price', 'volume', 'contract'],
                     frequency='daily', start_date='2017-06-01', end_date='2017-08-01')
ngf20_data.head(); ngf20_data.plot()  # WORK WITH THIS DATAFRAME HOWEVER
# F - January, G - February, ...
ng_contracts = symbols(['NGF18', 'NGG18', 'NGH18', 'NGJ18', 'NGK18', 'NGM18'])

#   FIRST, WORKING WITH CONSECUTIVE FUTURES
ng_consecutive_contract_volume = history(ng_contracts, fields='volume', frequency='daily',
                                         start_date='2017-08-01', end_date='2017-08-01')
ng_consecutive_contract_volume.plot()  # ANOTHER DATAFRAME :)
# VOLUME BEING TRADED INCREASES FUTURE'S PRICE AS IT ALSO INCREASES
# BUT ALSO, AS THE FUTURE'S MATURITY DATE APPROACHES, IT'S PRICE BEGINS TO REDUCE GRADUALLY

#   NOW, WORKING WITH CONTINUOUS FUTURES
ng_continuous = continuous_future('NG', offset=0, roll='volume', adjustment='mul')  # / 'add' / 'nul'
ng_continuous_active = history(ng_continuous, fields=['contract', 'price', 'volume'],
                               frequency='daily', start_date='2018-10-01', end_date='2019-08-01')
ng_continuous_active.head(); ng_continuous_active['contract'].unique()
ng_continuous_active['price'].plot(); ng_continuous_active['volume'].plot();
