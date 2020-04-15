from scipy import stats
from scipy.optimize import fsolve
import numpy as np
# t_maturity in years

def calc_price(volatility, spot_price, strike_price, t_maturity, interest_rate=0, option_type="call"):
    d_1 = (np.log(spot_price/strike_price) + (interest_rate + (volatility**2)/2)*t_maturity)/(volatility*np.sqrt(t_maturity))
    d_2 = d_1 - volatility*np.sqrt(t_maturity)

    if option_type == "call":
        n_d1 = stats.norm.cdf(d_1)
        n_d2 = stats.norm.cdf(d_2)
        return spot_price*n_d1 - strike_price*np.exp(-interest_rate*t_maturity)*n_d2
    else:
        n_d1 = stats.norm.cdf(-d_1)
        n_d2 = stats.norm.cdf(-d_2)
        return strike_price*np.exp(-interest_rate*t_maturity)*n_d2 - spot_price*n_d1


def calc_volatility(price, spot_price, strike_price, t_maturity, interest_rate=0, option_type="call"):
    def vega(vol):
        d_1 = (np.log(spot_price / strike_price) + (interest_rate + (vol ** 2) / 2) * t_maturity) / (vol * np.sqrt(t_maturity))
        return spot_price*stats.norm.pdf(d_1)*np.sqrt(t_maturity)
    imp_volatility = fsolve(lambda x: (calc_price(x, spot_price, strike_price, t_maturity, interest_rate, option_type) - price),
                            x0=0.27, fprime=vega, maxfev=2500)
    return imp_volatility[0]*100
