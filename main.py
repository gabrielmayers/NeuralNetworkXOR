import numpy as np

# Rede neural com operador XOR:

entradas = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
saidas = np.array([0, 1, 0, 1])
pesos = np.array([0.0, 0.0])
taxaAprendizagem = 0.1


# Função de ativação:
def stepFunction(soma):
    if soma >= 1:
        return 1
    return 0


# registros = entradas:

def calculaSaida(registro):
    # Produto das entradas pelos pesos:
    s = registro.dot(pesos)
    return stepFunction(s)


def treinar():
    # Inicializa-se o erro total com 1 para que ele possa entrar no loop treinamento:
    erroTotal = 1
    while erroTotal != 0:
        erroTotal = 0
        # for para percorrer todos os registros:
        for i in range(len(saidas)):
            # Calculando saida:
            saidaCalculada = calculaSaida(entradas[i])
            # abs vai tornar indiferente a ordem dos fatores para chegar num resultado:
            erro = abs(saidas[i] - saidaCalculada)
            # Incrementando o erroCalculado pelo erroTotal:
            erroTotal += erro
            # Atualizando os pesos:
            for j in range(len(pesos)):
                pesos[j] = pesos[j] + (taxaAprendizagem * entradas[i][j] * erro)
                print("Pesos atualizados: " + str(pesos[j]))
            print("Total de erros: " + str(erroTotal))


treinar()
print("Rede Neural treinada!")
print(calculaSaida(entradas[0]))
print(calculaSaida(entradas[1]))
print(calculaSaida(entradas[2]))
print(calculaSaida(entradas[3]))
