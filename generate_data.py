import pandas as pd

def gendata():

		df=pd.read_csv('dataset/fer2013/fer2013.csv')
		m = [i for i in range(35000)]

		dataframe = pd.DataFrame(index=m)

		for i in range(35000):
			l =[]
			l = df['pixels'][i].split(' ')
			for _ in range(len(l)):
				dataframe.loc[i,'pixel_'+str(_)] = int(l[_])
    

		for i in range(35000):
			dataframe.loc[i,'emotions']= df.loc[i,'emotion']

		dataframe.to_csv('emotions_df.csv')

gendata()