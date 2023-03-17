import pandas as pd

read_file = pd.read_csv (r'/Users/thomasliu/Catan-AI-1/code/data_medium.txt')
read_file.to_csv (r'/Users/thomasliu/Catan-AI-1/code/data_medium.csv', index=False, header=True)