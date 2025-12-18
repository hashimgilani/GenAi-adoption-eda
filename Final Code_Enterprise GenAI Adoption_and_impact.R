# === Load Required Packages and Dataset ===
library(tidyverse)
library(car)
library(ggplot2)
library(corrplot)
library(dplyr)
library(skimr)
library(readr)


Gen <- read.csv("Enterprise_GenAI_Adoption_Impact.csv")



names(Gen) <- c(
  "company",
  "industry",
  "country",
  "gen_tool",
  "adoption_year",
  "num_employees_impacted",
  "new_roles_created",
  "training_hours",
  "productivity_change",
  "employee_sentiment")

head(Gen)

# === L
# Apply controlled sampling for consistent modeling
set.seed(2025)
sampled_data <- sample_n(Gen, 500)

summary(select(Gen, adoption_year, productivity_change, training_hours, new_roles_created, num_employees_impacted))

ggplot(Gen, aes(x = productivity_change)) + 
  geom_histogram(binwidth = 1, fill = "skyblue", color = "black") +
  theme_minimal() +
  labs(title = "Distribution of Productivity Change")

ggplot(Gen, aes(x = adoption_year)) + 
  geom_histogram(binwidth = 1, fill = "skyblue", color = "black") +
  theme_minimal() +
  labs(title = "Distribution of adoption year")

ggplot(Gen, aes(x = training_hours)) + 
  geom_histogram(binwidth = 100, fill = "skyblue", color = "black") +
  theme_minimal() +
  labs(title = "Distribution of Training Hours",
       x = "Training Hours",
       y = "Count")

ggplot(Gen, aes(x = new_roles_created)) + 
  geom_histogram(binwidth = 1, fill = "skyblue", color = "black") +
  theme_minimal() +
  labs(title = "Distribution of new roles created")

ggplot(Gen, aes(x = num_employees_impacted)) + 
  geom_histogram(binwidth = 100, fill = "skyblue", color = "black") +
  theme_minimal() +
  labs(title = "Distribution of number of empolyees impacted")


summary(Gen[, c("training_hours", "num_employees_impacted")])

ggplot(Gen, aes(x = industry)) +
  geom_bar(fill = "skyblue") +
  theme_minimal() +
  labs(title = "Count by Industry", x = "Industry", y = "Count") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggplot(Gen, aes(x = country)) +
  geom_bar(fill = "skyblue") +
  theme_minimal() +
  labs(title = "Count by Country", x = "Country", y = "Count") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggplot(Gen, aes(x = gen_tool)) +
  geom_bar(fill = "skyblue") +
  theme_minimal() +
  labs(title = "Count by Generative Tool", x = "Gen Tool", y = "Count") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


num_vars <- c("adoption_year", "productivity_change", "training_hours", "new_roles_created", "num_employees_impacted")

cor_mat <- cor(Gen[, num_vars], use = "complete.obs")
corrplot(cor_mat, method = "color", addCoef.col = "black", number.cex = 0.8, tl.cex = 1, tl.srt = 45)

cor_matrix <- cor(Gen[, num_vars], use = "complete.obs")

print(round(cor_matrix, 3))


cat_vars <- c("company", "industry", "country", "gen_tool")
Gen[cat_vars] <- lapply(Gen[cat_vars], factor)


model <- lm(productivity_change ~ adoption_year + training_hours + new_roles_created +
            num_employees_impacted + industry + country + gen_tool, data = sampled_data)

summary(model)


residuals <- resid(model)

summary(residuals)

plot(model$fitted.values, model$residuals,
     main = "Residuals vs Fitted",
     xlab = "Fitted Values", ylab = "Residuals", col = "black")
abline(h = 0, col = "red", lwd = 2)

qqnorm(resid(model))
qqline(resid(model), col = "red", lwd = 2)


new_data <- Gen[1:5, ]


predictions <- predict(model, newdata = new_data)

results <- data.frame(
  Actual = new_data$productivity_change,
  Predicted = predictions,
  Error = new_data$productivity_change - predictions)

print(results)

mae <- mean(abs(results$Error))
mse <- mean(results$Error^2)

cat("MAE:", mae, "\nMSE:", mse, "\n")


back_model <- lm(productivity_change ~ training_hours + new_roles_created +
            num_employees_impacted + industry + gen_tool, data = sampled_data)

back_model <- step(back_model, direction = 'backward', trace = TRUE)



summary(back_model)



# 1) Define the null (intercept‐only) model
null_mod <- lm(productivity_change ~ 1, data = sampled_data)

# 2) Define the full formula (all candidates)
full_form <- productivity_change ~ training_hours +
              new_roles_created + num_employees_impacted +
              industry + gen_tool

# 3) Run forward stepwise
forward_model <- step(null_mod,
                      scope    = full_form,
                      direction = "forward",
                      trace     = TRUE)



summary(forward_model)


# Both‐directions stepwise
both_model <- step(null_mod,
                   scope     = full_form,
                   direction = "both",
                   trace     = TRUE)

summary(both_model)



# STEP 9: Final Model Summary and Residual Diagnostics (using sample)

# 1. Prepare sample data
set.seed(2025)
Gen_sampled <- dplyr::sample_n(Gen, 500)

# 2. Fit final model on sampled data (backward selection or specified predictors)
final_model <- lm(productivity_change ~ training_hours + new_roles_created +
                    num_employees_impacted + industry + gen_tool, data = Gen_sampled)

# 3. Print model summary
cat("\n Final Model Summary (on Sampled Data) \n")
summary(final_model)

# 4. VIF Check (if enough predictors)
model_terms <- attr(terms(final_model), "term.labels")
if (length(model_terms) >= 2) {
  library(car)
  vif_values <- vif(final_model)
  vif_df <- data.frame(
    Predictor = names(vif_values),
    VIF = round(as.numeric(vif_values), 2)
  )
  print(" VIF Values ")
  print(vif_df)
} else {
  cat("VIF not calculated: Final model has fewer than 2 predictor terms.\n")
}

# 5. Residuals vs Fitted Plot with LOESS Trend
residual_plot_data <- data.frame(
  Fitted = fitted(final_model),
  Residuals = resid(final_model)
)

ggplot(residual_plot_data, aes(x = Fitted, y = Residuals)) +
  geom_point(alpha = 0.4, color = "steelblue") +
  geom_smooth(method = "loess", se = FALSE, span = 1, color = "darkred", linewidth = 1.2, na.rm = TRUE) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  labs(
    title = "Final Model: Residuals vs Fitted Values (with LOESS Curve)",
    subtitle = "Flat LOESS line suggests linearity and good model fit",
    x = "Fitted Productivity Change (%)",
    y = "Residuals"
  ) +
  theme_minimal()

# 6. Accuracy Metrics (MAE and MSE)
predicted_vals <- predict(final_model, newdata = Gen_sampled)
actual_vals <- Gen_sampled$productivity_change

mae <- mean(abs(predicted_vals - actual_vals))
mse <- mean((predicted_vals - actual_vals)^2)

cat("\n Model Accuracy Metrics \n")
cat(paste("MAE (Mean Absolute Error):", round(mae, 2), "\n"))
cat(paste("MSE (Mean Squared Error):", round(mse, 2), "\n"))

