import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error # importamos el cálculo del error cuadrático medio (MSE)


def upload_data():
	spo = pd.read_csv('https://raw.githubusercontent.com/JHON-EDV/Regresion-Polinomica/master/datos_regresion.csv',names=["x1","y1","y1r"])	
	print("Los datos estan cargados")
	return spo

def regresion(spo):
	regression = []
	error1 = []
	error2 = []
	degreee = []
	# Se recorre con un for de números inpares.
	for degree in range(3, 20, 2):

	  # Importamos la clase de Regresión Lineal de scikit-learn
	  from sklearn.linear_model import LinearRegression 
	  # para generar características polinómicas
	  from sklearn.preprocessing import PolynomialFeatures 
	  pf = PolynomialFeatures(degree)    # usaremos polinomios de grado 3
	  X  = pf.fit_transform(spo['x1'].values.reshape(-1,1))
	  regresion_lineal = LinearRegression() # creamos una instancia de LinearRegression
	  # instruimos a la regresión lineal que aprenda de los datos (ahora polinómicos) (X,y)
	  regresion_lineal.fit(X, spo['y1r']) 
	  # Guardamos en un array los datos de el polinomio obtenido

	  items = []
	  for x in range(degree):
	    #print(x)
	    if not x:
	        items.append('{:.4f}'.format(regresion_lineal.intercept_))
	        continue
	        
	    items.append('{:.4f}x^{:.0f}'.format(regresion_lineal.coef_[x] if regresion_lineal.coef_[x] != 1 else '', x))
	    
	    result = ' + '.join(items)
	    result = result.replace('x^0', '')
	    result = result.replace('^1 ', ' ')
	    result = result.replace('+ -', '- ')
	  
	  regression.append(result)
	  
	  # Predecimos los valores y para los datos usados en el entrenamiento
	  prediccion_entrenamiento = regresion_lineal.predict(X)
	  # Calculamos el Error de Entrenamiento o de Estimacion 
	  error_entrenamiento = np.sum(np.power(np.subtract(spo['y1r'], prediccion_entrenamiento),2))
	  #print("El error de entrenamiento es {:.8f}".format(error_entrenamiento))
	  # Calculamos el Error de Entre funciones.
	  error_funciones = integrate.simps(np.power(np.subtract(spo['y1r'], prediccion_entrenamiento),2))

	  #Guardamos los resultados.
	  error1.append(error_entrenamiento)
	  error2.append(error_funciones)
	  degreee.append(degree)

  	


  	# Se agregan los resultados a un data frame.
	rounded_df = pd.DataFrame(list(zip(degreee, regression, error1, error2)),
               columns =['Grado de regresión','Polinomio', 'Error de entrenamiento', 'Error de estimación'])
	print("La regresión esta hecha")
	return rounded_df

def grafica_tabla(rounded_df): 
	# se presentan los datos en una tabla.
	fig =plt.figure(constrained_layout= False, figsize=(1,1) ,facecolor='w', edgecolor='k', dpi=180)
	ax = fig.add_subplot(111)
	column_labels=['Grado de regresión','Polinomio', 'Error de entrenamiento', 'Error de estimación']
	ax.axis('tight')
	ax.axis('off')
	the_table = plt.table(cellText= rounded_df.values, colLabels=column_labels, loc='center left')

	the_table.auto_set_font_size(False)
	the_table.set_fontsize(10)
	the_table.scale(1.5, 1.5)
	the_table.auto_set_column_width(col=list(range(len(rounded_df.columns)))) # Provide integer list of columns to adjust
	#plt.show()
	plt.savefig("Tabla de resultados con ruido.svg", bbox_inches='tight')
	print("los datos estan en una tabla")



def main():
	spo = upload_data()
	rounded_df = regresion(spo)
	grafica_tabla(rounded_df)




if __name__ == '__main__':
	main()


		