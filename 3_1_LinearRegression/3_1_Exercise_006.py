
'''
3_1 Exercise 006
'''

# We use names that will not interfere with the previous ones

df_07 = pd.read_csv("./3_1_data07.csv")
print(df_07.head(13))

model07_Formula = "y ~ x1 + C(x2)"
Y07, X07 = ps.dmatrices(model07_Formula, df_07)
X07[:13]
