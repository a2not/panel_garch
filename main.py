# % VT Estimation: Panel GARCH
# % Monte Carlo Experiment
from panel_garch import panel_garch
import pandas

df = pandas.read_excel("inflation_ratesG7.xlsx", sheet_name="Sheet1").iloc[:, 1:].values

pg = panel_garch(dataframe=df, iR=5)
# pg = panel_garch(dataframe=df, iR = 100)
pg.run(DGP=False, debug_print=True)
# pg.run(debug_print=True)

