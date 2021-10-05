library(class)
library(ISLR)
library(boot)
library(stringi)

#############################
# Pto 1: MODELOS DE REGRESIÓN
#############################

# En el mercado del real estate, el precio de los inmuebles es una variable crucial a la 
# hora de realizar una operación. Conocer el valor de mercado de una propiedad es clave 
# tanto para comprarla y venderla como así también para alquilarla. Este datasetr 
# permite tasar un inmueble para conocer los valores en los que se mueve el mercado 
# o averiguar el precio promedio en la zona en la que estás buscando mudarte.


# Cargar dataset "propiedades.csv" con función read.csv

# a) Ver estructura del dataset

# b) Seleccionar propiedades cuyo precio sea mayor a U$S50.000 y menor a U$S800.000

# c) Entrenar un modelo de regresión lineal simple para predecir la variable precio 
# para cada una las variables predictoras numéricas

# d) Indicar una medida de  performance predictiva para cada modelo utilizando validación cruzada 5K-CV, 
# presentando un dataset con los resultados de cada fold para cada modelo. 


#a

data <- read.csv('C:/Users/elbar/OneDrive/Documentos/ingenieria/modelos/datasets/propiedades.csv')

str(data)

View(data)

#b

grupo <- data[ (data$precio > 50000)&(data$precio<800000) , ] 
grupo

#c

lm.fit.cantAmbs <- lm(precio~cant_amb,data=data)
lm.fit.cantAmbs

lm.fit.cantBans <- lm(precio~cant_banos,data=data)
lm.fit.cantBans

lm.fit.supTotal <- lm(precio~sup_total,data=data)
lm.fit.supTotal

lm.fit.supCub <- lm(precio~sup_cub,data=data)
lm.fit.supCub


#d

data <- data[, c("cant_amb", "cant_banos","sup_total", "sup_cub","precio")]
View(data)
variables <- names(data)
variables <- variables[-5]

df.final <- data.frame()

for(v in variables){
  
  form <- paste0("precio~",v)
  ik <- 1
  nrow(data)
  tamfold <- as.integer(0.8*nrow(data))
  restof <- as.integer(nrow(data) - tamfold) 
  
  while( ik <= 5){
    
    inicio <- ((ik*restof) - restof)+1
    
    #fin <- inicio + tamfold 
    
    if(ik==5){
      fin <- inicio + restof -1
    }else{
      fin <- inicio + restof
    }
    
    #trainf <- data[inicio:fin,]
    trainf <- data[-(inicio:fin),]
    trainf
    #testf <- data[-(inicio:fin),]
    testf <- data[(inicio:fin),]
    
    lm.fitf <- lm(as.formula(form), data = trainf) 
    
    predsf <- predict(lm.fitf, newdata = testf[-5])
    
    mse<- 0
    observaciones <- c(testf[5])
    observaciones <- unlist(observaciones, use.names=FALSE)
    pos <- 1
    while(pos < length((predsf))){
      mse<- mse + (as.numeric(I((observaciones[pos] - predsf[pos])^2)))
      pos <- pos+1
    }
    mse <- mse/length(predsf)
    mse
    
    info.fold <- c(v,ik,mse)
    df.final <- rbind(df.final, info.fold)
    
    ik <- ik +1
  }
}

names(df.final) <- c("variables", "k-fold", "MSE")
df.final

#################################
# Pto 1: MODELOS DE CLASIFICACION
#################################
# El dataset mark_banco.csv registra datos reales de campañas de telemarketing para vender depósitos 
# a plazo fijo llevadas a cabo por banco minorista portugués. 
# Dentro de una campaña, los operadores ofrecen los productos y/o servicios de dos maneras y 
# en momentos diferentes: realizando llamadas telefónicas (saliente) o 
# aprovechando la llamada al call center de un cliente por cualquier otra razón (entrante). 
# Por lo tanto, el resultado es un contacto de resultado binario: éxito/positivo (compra el producto) 
# o fracaso/negativo (no lo compra).
# Los datos corresponden a contactos con clientes realizados mayo de 2008 hasta junio de 2013 y 
# totalizan  35.010 registros. Cada registro incluye el resultado del contacto en la cariable "y"
# y diferentes variables predictoras o atributos. Estos incluyen atributos de telemarketing 
# ,detalles del producto (por ejemplo tasa de interés ofrecida) e información del cliente 
# (por ejemplo edad). Estos registros se enriquecieron con datos de 
# índole de socio-económica (por ejemplo la tasa de desempleo). 
# 

# a) Ver estructura del dataset
#
# b) Se pide:
#   i) crear un conjunto de entrenamiento y otro de prueba con el dataset

#   ii) Entrenar un modelo de Regresion Logistica utilizando todas las variables sobre el
#      conjunto de entrenamiento.

#   iii) Para umbrales entre 0.1 y 0.9 (con pasos de a 0.1), calcular la exactitud, precision y 
#       recall/sensitividad sobre el conjunto de prueba. 

#   iv) Graficar la exactitud con respecto al umbral para el rango entre 0.1 y 0.9 con los valores 
#     obtenidos. ¿Como se puede explicar el comportamiento de la curva en base a la presencia de FN y FP?


#a

data2 <- read.csv2('C:/Users/elbar/OneDrive/Documentos/ingenieria/modelos/datasets/mark_banco.csv')

str(data2)

#b

data2$job <- as.numeric(as.factor(data2$job))
data2$marital <- as.numeric(as.factor(data2$marital))
data2$education <- as.numeric(as.factor(data2$education))
data2$default <- as.numeric(as.factor(data2$default))
data2$housing <- as.numeric(as.factor(data2$housing))
data2$loan <- as.numeric(as.factor(data2$loan))
data2$contact <- as.numeric(as.factor(data2$contact))
data2$month <- as.numeric(as.factor(data2$month))
data2$day_of_week <- as.numeric(as.factor(data2$day_of_week))
data2$poutcome <- as.numeric(as.factor(data2$poutcome))
data2$y = as.factor(data2$y)
levels(data2$y) = c(0,1)

set.seed(123)
randoms <- sample(nrow(data2), 0.8*nrow(data2))

train <- data2[randoms,] #i

test <- data2[-randoms,]

classifier <- glm(y~., family = binomial, data = train) #ii

coef(classifier)
View(data2)

preds_glm <- predict(classifier, type = 'response', newdata = test[-21])

preds_glm

umbral <- 0.1

datos.totales <- data.frame()

while(umbral<=0.9){
  
  y_preds <- ifelse(preds_glm > umbral, 1, 0)
  
  y_preds
  
  matc <- table(y_preds, test$y, dnn = c("Pred" , "Real")) 
  
  matc
  
  FN <- matc[1,2]
  FN
  FP <- matc[2,1]
  FP
  VP <- matc[2,2]
  VP
  VN <- matc[1,1]
  VN
  
  accuracy <- (VP+VN) / (VP+VN+FN+FP)
  accuracy
  
  prec <- VP / (VP + FP)
  prec
  
  mem <- VP / (VP + FN)
  mem
  
  datos.iteracion <- c(umbral,accuracy,prec,mem)
  datos.totales <- rbind(datos.totales, datos.iteracion)
  umbral <- umbral + 0.1
}

names(datos.totales) <- c("umbral", "accuracy", "precision", "memoria")
datos.totales #iii

plot(datos.totales$umbral, datos.totales$accuracy) #iiii

# En el grafico podemos apreciar como cuanto mas nos acercamos a los extremos aumenta la cantidad de falsos
# en nuestras predicciones, cuanto mas alto pongamos el umbral, mas controlaremos los falsos positivos pero
# aumentaran considerablemente los falsos negativos. Al reducir el umbral pasa exactamente lo contrario,
# por lo tanto en el grafico podemos apreciar como sucede esto al dismunir la exactitud hacia los extremos
# y aumentando en el centro.