# Librerie necessarie
library(tidyverse)
library(skimr)
library(caret)
library(randomForest)
library(gbm)
library(dplyr)
library(ggplot2)
library(reshape2)
library(cvms)
library(kableExtra)
library(knitr)
library(gridExtra)
library(grid)

# Carico dataset 
diabetes_data <- read.csv("C:\\Users\\totim\\Downloads\\Data_mining_appello_7_aprile_2025 (1)\\diabetes_data.csv", sep = ";")

# Controllo struttura, valori mancanti e  Statistiche descrittive generali
str(diabetes_data)
summary(diabetes_data)
colSums(is.na(diabetes_data))
# visualizzo i nomi delle colonne del dataset ed  elimino prima colonna con natura dei dati non specificata
colnames(diabetes_data)
diabetes_data <- diabetes_data %>% select(-X)
# Calcolo numero valori BMI sopra 120
num_bmi_over_150 <- sum(diabetes_data$BMI > 120, na.rm = TRUE)

# Visualizzo il risultato
print(paste("Numero di valori di BMI superiori a 120:", num_bmi_over_150))

# Calcolo il numero di valori negativi di LDL e HDL
sum((diabetes_data[, c("LDL", "HDL")]) < 0)

# rumuovo righe con valori anomali, il totale delle osservazioni che presentano valori anomali
# è minore delle 0,26% delle osservazioni nel dataset, questo mi consente di semplificare la trattazione,
# mantenendo comunque la robustezza dell'analisi 
diabetes_data <- diabetes_data %>% 
  filter(LDL >= 0, HDL >= 0, BMI <= 120)



# Distribuzione esito diabete
table(diabetes_data$Outcome)
prop.table(table(diabetes_data$Outcome))



# Creo il bar plot della variabile Outcome (diabete) e lo assegno  ad un oggetto
grafico_outcome <- ggplot(diabetes_data, aes(x = factor(Outcome))) +
  geom_bar(fill = c("lightblue", "salmon")) +
  labs(
    x = "Outcome (0 = No Diabete, 1 = Diabete)",
    y = "Numero di casi"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 18, face = "bold"),
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 12)
  )

# Mostro il grafico a schermo
print(grafico_outcome)

# Salvo il grafico in alta definizione
ggsave("distribuzione_outcome1.png", plot = grafico_outcome,
       width = 8, height = 6, dpi = 1000)

# Creo il boxplot delle variabili quantitative condizionate all'Outcome  e lo salvo in un oggetto
boxplot_outcome <- diabetes_data %>%
  pivot_longer(cols = c(Age, BMI, Glucose, HbA1c, LDL, HDL, BloodPressure, WaistCircumference, WHR),
               names_to = "Variabile", values_to = "Valore") %>%
  ggplot(aes(x = factor(Outcome), y = Valore, fill = factor(Outcome))) +
  geom_boxplot() +
  facet_wrap(~ Variabile, scales = "free") +
  labs(x = "Outcome", y = "Valore") +
  theme_minimal() +
  theme(
    legend.position = "none",
    strip.text = element_text(size = 20),
    axis.title = element_text(size = 18),
    axis.text = element_text(size = 11),
    plot.title = element_text(size = 25)
  )

# stampo il grafico a schermo
print(boxplot_outcome)

# salvo il grafico il alta definizione
ggsave("boxplot_variabili_numeriche_.png", plot = boxplot_outcome,
       width = 14, height = 10, dpi = 1000)





# lista delle variabili categoriali
categorical_vars <- c("FamilyHistory", "DietType", "Hypertension", "MedicationUse")
plots <- list()

# Creo i barplot delle variabili categoriali condizionate all'outcome 
for (var in categorical_vars) {
  p <- ggplot(diabetes_data, aes_string(x = var, fill = "factor(Outcome)")) +
    geom_bar(position = "fill") +
    scale_y_continuous(labels = scales::percent) +
    labs(title = paste("Distribuzione di", var, "\ncondizionata a Outcome"),
         x = var, y = "Frequenza relativa",
         fill = "Outcome (0 = No Diabete, 1 = Diabete)") +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 19, hjust = 0.5),  
      axis.title.x = element_text(size = 17),
      axis.title.y = element_text(size = 16),
      axis.text.x  = element_text(size = 13),
      axis.text.y  = element_text(size = 14),
      legend.position = "bottom",
      legend.text = element_text(size = 14),
      legend.title = element_text(size = 14)
    )
  plots[[var]] <- p
}



# Creo il layout con grid.arrange e lo salvo come oggetto
grafico_unico <- arrangeGrob(grobs = plots, ncol = 2)

# stampo grafico a schermo
grid.newpage()
grid.draw(grafico_unico)


# Salvo grafico in alta  qualità
ggsave("grafici_categoriali1_.png", plot = grafico_unico, width = 14, height = 10, dpi = 1000)

# Calcolo percentuale di pazienti con assenza di familiarità che presentano diabete

no_family_history <- diabetes_data[diabetes_data$FamilyHistory == "No storia familiare", ]

percentuale_diabetici <- mean(no_family_history$Outcome == "Diabetico") * 100

# Stampo il risultato
cat("Percentuale di diabetici senza familiarità:", round(percentuale_diabetici, 2), "%\n")

# Seleziono solo le variabili numeriche (escludendo quelle categoriche)
numeric_vars <- select(diabetes_data, where(is.numeric))


# Calcolo la matrice delle correlazioni
cor_matrix <- cor(numeric_vars, use = "pairwise.complete.obs")

# Converto la matrice in un data frame lungo
melted_cor_matrix <- melt(cor_matrix)

# Creo la heatmap con i valori delle correlazioni
ggplot(data = melted_cor_matrix, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(color = "white") +  # Caselle colorate
  geom_text(aes(label = round(value, 2)), color = "black", size = 4) +  # Aggiunge i valori di correlazione
  scale_fill_gradient2(low = "blue", high = "red", mid = "white",
                       midpoint = 0, limit = c(-1,1), space = "Lab",
                       name = "Correlazione") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1)) +
  coord_fixed() +
  labs(title = "Matrice delle Correlazioni", x = "", y = "")



# RANDOM FOREST 
# creo un dataset per l'addestramento del modello ed uno per il test, ognuno rispettivamente con il 75% e il 25% delle osservazioni 

set.seed(123)
sel <- sample(1:nrow(diabetes_data), size = floor(0.75 * nrow(diabetes_data)), replace = FALSE)

# Converto la variabile Outcome in fattore (con livelli "0" e "1") perché Random Forest in R 
# tratta come regressione i target numerici, mentre noi vogliamo fare classificazione binaria.

diabetes_data$Outcome <- as.factor(diabetes_data$Outcome)
train <- diabetes_data[sel, ]
test <- diabetes_data[-sel, ]
# mi assicuro che i positivi e i negativi al diabete  mantengano nei dataset più o meno la stessa proporzione 
table(diabetes_data$Outcome)
table(test$Outcome)
table(train$Outcome)

# Tratto lo sbilanciamento delle classi, attribuendo agli errori di previsione dell'outcome positivo un peso 
# pari al rapporto tra negativi e positivi nel dataset di addestramento
p0 <- sum(train$Outcome == "0")
p1 <- sum(train$Outcome == "1")
class_weights <- c("0" = 1, "1" = p0 / p1)

# In questo blocco costruisco nove modelli Random Forest, combinando diversi valori di ntree (numero di alberi) e mtry (numero di variabili candidate a ogni split). 
# Per ciascun modello calcolo le previsioni probabilistiche ed esploro un range di soglie di classificazione tra 0.10 e 0.50, con un passo di 0,05.
# successivamente calcolo le principali metriche di valutazione per ogni soglia, e seleziono la soglia ottimale in base al massimo valore di F1-score.
# Questo mi consente di confrontare le diverse configurazioni in modo più robusto, tenendo conto dell'importanza della sensibilità in contesti clinici.
# Escludo la variabile FamilyHistory da entrambi i tipi di modelli perchè permetteva previsioni troppo semplici, andando ad intaccare lo scopo didattico del report. 

# Combinazioni di ntree e mtry
tree_vals <- c(500, 1000, 1500)
mtry_vals <- c(2, 3, 4)

rf_models <- list()

# Addestramento modelli e salvataggio
for (ntree in tree_vals) {
  for (mtry in mtry_vals) {
    model_name <- paste0("rf_", ntree, "_", mtry)
    model <- randomForest(
      Outcome ~ .- FamilyHistory, data = train, ntree = ntree, mtry = mtry,
      importance = TRUE, classwt = class_weights
    )
    rf_models[[model_name]] <- model
  }
}
# Calcolo metriche per tutti i modelli RF con ottimizzazione della soglia per F1
thresholds <- seq(0.1, 0.5, by = 0.05)

metrics_list <- lapply(names(rf_models), function(model_name) {
  model <- rf_models[[model_name]]
  
  # Previsioni probabilistiche
  pred_probs <- predict(model, newdata = test, type = "prob")[, "1"]
  
  # Calcolo metriche per diverse soglie
  metriche_per_soglia <- lapply(thresholds, function(thresh) {
    pred_class <- ifelse(pred_probs > thresh, 1, 0)
    cm <- confusionMatrix(factor(pred_class, levels = c(0, 1)),test$Outcome, positive = "1")
    
    acc <- cm$overall["Accuracy"]
    sens <- cm$byClass["Sensitivity"]
    prec <- cm$byClass["Pos Pred Value"]
    f1 <- if ((prec + sens) == 0) 0 else 2 * (prec * sens) / (prec + sens)
    
    data.frame(
      Threshold = thresh,
      Accuracy = acc,
      Sensitivity = sens,
      Precision = prec,
      F1 = f1
    )
  })
  
  # Unisco i risultati e prendo la soglia con F1 più alta
  metriche_df <- do.call(rbind, metriche_per_soglia)
  best_row <- metriche_df[which.max(metriche_df$F1), ]
  
  # Output finale per questo modello
  data.frame(
    Modello = model_name,
    Threshold = best_row$Threshold,
    Accuracy = best_row$Accuracy,
    Sensitivity = best_row$Sensitivity,
    Precision = best_row$Precision,
    F1 = best_row$F1
  )
})

# Combino in una tabella
metrics_df <- do.call(rbind, metrics_list)

# Ordino per F1 decrescente
metrics_df_sorted <- metrics_df[order(-metrics_df$F1), ]

# Visualizzo la tabella con tutti gli iperparametri dei modelli migliori di Random Forest.
kable(metrics_df_sorted, format = "html", digits = 3, caption = "Metriche Random Forest", row.names = FALSE) %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"), full_width = F)

# Estraggo il nome e la soglia ottimale del miglior modello RF
best_rf_name <- metrics_df_sorted$Modello[1]
best_rf_model <- rf_models[[best_rf_name]]
best_rf_threshold <- metrics_df_sorted$Threshold[1]

# Previsioni probabilistiche e classificazione con soglia ottimale
pred_probs_rf <- predict(best_rf_model, newdata = test, type = "prob")[, "1"]
pred_class_rf <- ifelse(pred_probs_rf > best_rf_threshold, 1, 0)

# Matrice di confusione, questa verrà esclusa dal report finale per evitare di riportare informazioni ridondanti,
# essendo già presenti le metriche di sensibilità e precisione 
cm_rf <- confusionMatrix(as.factor(pred_class_rf), as.factor(test$Outcome), positive = "1")

# Converto per la visualizzazione
cfs_rf <- as.data.frame(cm_rf$table)

# Visualizzo la matrice
plot_confusion_matrix(cfs_rf,
                      target_col = "Reference",
                      prediction_col = "Prediction",
                      counts_col = "Freq",
                      add_normalized = FALSE,
                      place_x_axis_above = FALSE)

# Imposto margini più larghi per far spazio alle etichette
par(mar = c(8, 4, 4, 2))  


# Importanza variabili - Random Forest
var_import_best <- best_rf_model$importance[, "MeanDecreaseGini"]
var_import_sorted_best <- sort(var_import_best, decreasing = TRUE)

barplot(var_import_sorted_best[1:10],
        main = paste("Importanza variabili - Modello", best_rf_name),
        col = rainbow(10),
        cex.names = 0.8,
        las = 2)


# GRADIENT BOOSTING

# Prima di procedere con l’ottimizzazione degli iperparametri, ho costruito un primo modello di Gradient Boosting. 
# Ho impostato un numero elevato di alberi (n.trees = 1000) e ho attivato 
# la cross-validation a 5 fold per valutare le performance in modo robusto. Successivamente, ho usato la funzione 
# gbm.perf() per analizzare l’andamento della devianza di Bernoulli, ovvero la funzione di perdita che misura 
# l’errore predittivo in problemi di classificazione binaria. Dal grafico prodotto, ho osservato che la devianza 
# tende a stabilizzarsi già dopo circa 500 alberi, segnalando che ulteriori alberi apportano benefici marginali. 
# Tuttavia ho mantenuto n.trees = 1000: un valore che garantisce buone prestazioni senza incorrere in overfitting,
# grazie alla regolazione  garantita dal parametro shrinkage e alla validazione incrociata.
# Il resto del codice segue la stessa logica del blocco sul Random Forest, variando la soglia (Threshold), per garantire l'ottimizzazione dell'iperparametro F1.


# Creo una copia con Outcome numerico solo per il boosting
train_gbm <- train
train_gbm$Outcome <- as.numeric(as.character(train_gbm$Outcome))  # Converte fattore in 0/1 numerico






# Costruzione del modello di Boosting
set.seed(123)  # Per riproducibilità
boost_model <- gbm(
  formula = Outcome ~ ., 
  data = train_gbm,
  distribution = "bernoulli",
  n.trees = 1000,                   # Numero massimo di alberi
  interaction.depth = 3,            # Profondità degli alberi 
  shrinkage = 0.01,                 # Tasso di apprendimento
  cv.folds = 5,                     # Numero di fold per la cross-validation
)

# Trovo il numero ottimale di alberi (minimizzo l'errore di CV)
best.iter <- gbm.perf(boost_model, method = "cv")




# MODELLI BOOSTING (GBM)


# Lista dei parametri da combinare
depth_vals <- c(2, 3, 4)
shrink_vals <- c(0.001, 0.005, 0.01)

# Lista dove salveremo i modelli
boosting_models <- list()

# Costruzione dei modelli con cicli nidificati
for (depth in depth_vals) {
  for (shrink in shrink_vals) {
    model_name <- paste0("gbm_d", depth, "_sh", gsub("\\.", "", as.character(shrink)))
    
    set.seed(123)  # Per riproducibilità
    model <- gbm(
      formula = Outcome ~ .- FamilyHistory, 
      data = train_gbm,
      distribution = "bernoulli",
      n.trees = 1000,                   # Numero massimo di alberi
      interaction.depth = depth,        # Profondità dell'albero
      shrinkage = shrink,               # Tasso di apprendimento
      cv.folds = 5,                     # Cross-validation 5-fold
    )
    
    # Salva il modello nella lista
    boosting_models[[model_name]] <- model
    
    # Mostra messaggio di completamento
    cat("Creato modello:", model_name, "\n")
  }
}

# Calcolo metriche per ogni modello di boosting salvato con soglia variabile
thresholds <- seq(0.1, 0.5, by = 0.05)

metrics_boosting <- lapply(names(boosting_models), function(model_name) {
  model <- boosting_models[[model_name]]
  pred_probs <- predict(model, newdata = test, n.trees = 1000, type = "response")
  
  # Provo più soglie
  metriche_per_soglia <- lapply(thresholds, function(thresh) {
    pred_class <- ifelse(pred_probs > thresh, 1, 0)
    cm <- confusionMatrix(factor(pred_class, levels = c(0, 1)),test$Outcome, positive = "1")
    
    acc <- cm$overall["Accuracy"]
    sens <- cm$byClass["Sensitivity"]
    prec <- cm$byClass["Pos Pred Value"]
    f1 <- if ((prec + sens) == 0) 0 else 2 * (prec * sens) / (prec + sens)
    
    data.frame(
      Soglia = thresh,
      Accuracy = acc,
      Sensitivity = sens,
      Precision = prec,
      F1 = f1
    )
  })
  
  # Unisco le metriche per soglia in un solo data frame
  metriche_df <- do.call(rbind, metriche_per_soglia)
  
  # Prendo la soglia con F1 più alta
  best_row <- metriche_df[which.max(metriche_df$F1), ]
  
  # Ritorna le metriche del modello con soglia ottimale
  data.frame(
    Modello = model_name,
    Threshold = best_row$Soglia,
    Accuracy = best_row$Accuracy,
    Sensitivity = best_row$Sensitivity,
    Precision = best_row$Precision,
    F1 = best_row$F1
  )
})


# Unisco le righe
metrics_boosting_df <- do.call(rbind, metrics_boosting)

# Ordino per F1 decrescente
metrics_boosting_df_sorted <- metrics_boosting_df[order(-metrics_boosting_df$F1), ]


# visualizzo tabella con iperparametri migliori orinati per F1 decrescenti
kable(metrics_boosting_df_sorted, format = "html", digits = 3,
      caption = "Metriche Modelli Boosting (GBM)", row.names = FALSE) %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"),
                full_width = FALSE)

# Prendo il nome del modello boosting con F1 più alta
best_boost_name <- metrics_boosting_df_sorted$Modello[1]

# Recupero il modello dalla lista
best_boost_model <- boosting_models[[best_boost_name]]

# Estraggo importanza delle variabili 
var_import_boost <- summary(best_boost_model, plotit = FALSE)

# Ordino in ordine decrescente
var_import_boost_sorted <- var_import_boost[order(-var_import_boost$rel.inf), ]


# Imposto margini più larghi per far spazio alle etichette
par(mar = c(10, 4, 4, 2))  


# Visualizzo le prime 10 variabili più importanti
barplot(
  var_import_boost_sorted$rel.inf[1:10],
  names.arg = var_import_boost_sorted$var[1:10],
  main = paste("Importanza variabili - Boosting", best_boost_name),
  col = rainbow(10),
  las = 2,
)
# Recupero la soglia ottimale del miglior modello
optimal_threshold <- metrics_boosting_df_sorted$Threshold[1]

# Previsioni probabilistiche e classificazione con soglia ottimale
pred_probs_boost <- predict(best_boost_model, newdata = test, n.trees = 1000, type = "response")
pred_class_boost <- ifelse(pred_probs_boost > optimal_threshold, 1, 0)

# Matrice di confusione 
cm_boost <- confusionMatrix(as.factor(pred_class_boost), as.factor(test$Outcome), positive = "1")

# Calcolo metriche
precision_boost <- cm_boost$byClass["Pos Pred Value"]
recall_boost <- cm_boost$byClass["Sensitivity"]
F1_boost <- 2 * (precision_boost * recall_boost) / (precision_boost + recall_boost)

# Converto in formato adatto per plot_confusion_matrix
cfs_boost <- as.data.frame(cm_boost$table)

# Visualizzazione grafica
plot_confusion_matrix(cfs_boost,
                      target_col = "Reference",        # Colonna con i valori reali
                      prediction_col = "Prediction",   # Colonna con le previsioni
                      counts_col = "Freq",             # Conteggi assoluti
                      add_normalized = FALSE,          
                      place_x_axis_above = FALSE)      # Asse X in basso
