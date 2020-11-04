# % VT Estimation: Panel GARCH
# % Monte Carlo Experiment
from src import panel_garch
import pandas

df = pandas.read_excel("inflation_ratesG7.xlsx",
                       sheet_name="Sheet1").iloc[:, 1:].values

pg = panel_garch(dataframe=df, iR = 100)
pg.run(DGP=True, debug_print=True)
