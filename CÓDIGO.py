        ###########################################
        #########   Qualidade do sono  ############
        ###########################################   


#    Em uma pesquisa, alguns dados foram coletados para a análise do sono em alguns indivíduos
# dentre os dados mais importantes, são eles:
# Horas de sono, Nível de ansiedade, Nível de Depressão e Uso de Luzes Artificiais.



######################    Importar bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


######################    Dados do modelo
#Ct, Horas de sono, Nível de ansiedade, Nível de Depressão e uso de luzes artificiais
X = np.array([    
[1, 8, 4, 3, 2],
[1, 4, 3, 1, 3],
[1, 6, 4, 3, 0],
[1, 9, 2, 0, 4],
[1, 8, 3, 0, 0],
[1, 6, 0, 0, 0],
[1, 7, 0, 0, 0],
[1, 5, 2, 1, 3],
[1, 6, 4, 2, 4]
])

#    Resultados a partir da média dos valores acima
Y = np.array([
[0],
[1],
[2],
[2],
[3],
[3],
[4],
[4],
[1]
])

#########################################


######################    Criação do modelo
Xt = np.transpose(X)          
XtX = np.matmul(Xt,X)         
XtX_inv = np.linalg.inv(XtX)  
XtY = np.matmul(Xt,Y)         
coef = np.matmul(XtX_inv,XtY) 
X_one = X

#coef
#print(coef)

###########################################


######################    Teste do modelo com o conjunto teste
#Ct, Horas de sono, Nível de ansiedade, Nível de Depressão e uso de luzes artificiais
X_teste=([
[1, 8, 2, 1, 4],
[1, 7, 4, 2, 1],
[1, 4, 4, 3, 0],
[1, 8, 2, 0, 0],
[1, 6, 1, 1, 4],
[1, 9, 2, 0, 0],
[1, 7, 2, 0, 0]
])

Yprev_teste = np.matmul(X_teste, coef)
#print(Yprev_teste)

##############################################################


######################    Texto pré-resultados
print('-' * 80)
print('#' * 25, '\033[1m' + 'Previsão da sua qualidade de sono') # \033[1m texto em bold
print('-' * 25, '\033[0m' + 'Parâmetros usados para análise:\n\
Horas de sono, Nível de ansiedade, Nível de Depressão e Uso de Luzes Artificiais\n') # \033[0m texto normal
print('-' * 25, 'Abaixo temos a tabela mostrando os resultados obtidos:')

##############################################


######################    Criando o DataFrame
Y_real = ([[2], [2], [2], [3], [3], [4], [3]])
df1 = pd.DataFrame(Yprev_teste) # Valores previstos pelo modelo para o conjunto teste
df2 = pd.DataFrame(Y_real) #Valores reais do teste

erro = df1 - df2 # Criar DataFrame com erro de previsão
df_concat = pd.concat([df1, df2, erro], axis = 1)  # Unindo dados no DataFrame
df_concat.columns = ['Previsto', 'Real', 'Erro'] # Rotulando as séries
df_concat

#############################################