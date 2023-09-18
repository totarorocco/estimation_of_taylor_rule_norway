###LOADING LIBRARIES###
if(system.file(package='ggplot2') == ""){
  install.packages('ggplot2')
  }
library("ggplot2")

if(system.file(package='dplyr') == ""){
  install.packages('dplyr')
  }
library("dplyr")

if(system.file(package='lmtest') == ""){
  install.packages('lmtest')
  }
library("lmtest")

if(system.file(package='tsoutliers') == ""){
  install.packages('tsoutliers')
}
library("tsoutliers")

if(system.file(package='readxl') == ""){
  install.packages('readxl')
}
library("readxl")

if(system.file(package='aTSA') == ""){
  install.packages('aTSA')
}
library("aTSA")

if(system.file(package='knitr') == ""){
  install.packages('knitr')
}
library("knitr")

if(system.file(package='dynlm') == ""){
  install.packages('dynlm')
}
library("dynlm")

if(system.file(package='car') == ""){
  install.packages('car')
}
library("car")

if(system.file(package='readxl') == ""){
  install.packages('readxl')
}
library("readxl")

if(system.file(package='tidyverse') == ""){
  install.packages('tidyverse')
}
library("tidyverse")

if(system.file(package='broom') == ""){
  install.packages('broom')
}
library("broom")

if(system.file(package='performance') == ""){
  install.packages('performance')
}
library("performance")

if(system.file(package='leaps') == ""){
  install.packages('leaps')
}
library("leaps")

if(system.file(package='glmnet') == ""){
  install.packages('glmnet')
}
library("glmnet")

###LOADING DATASET###
file_path <- file.choose()
df <- read_excel(file_path)

summary(df)


###PRELIMINARY VISUALIZATION###
#policy rate
df %>% ggplot(aes(time, rate)) + geom_point() + stat_smooth(method = "lm", formula = y ~ x, alpha = 0.2)
#GDP
df %>% ggplot(aes(time, gdp)) + geom_point() + stat_smooth(method = "lm", formula = y ~ x, alpha = 0.2)
#inflation
df %>% ggplot(aes(time, actual)) + geom_point() + stat_smooth(method = "lm", formula = y ~ x, alpha = 0.2)


###REGRESSION###

fit <- lm(data = df, rate ~ target + actual + gdp + potential)
summary(fit) #it is possible to see that target&actual, gdp&potential have inverse signs --> recalling the Taylor rule
#Everything is significant other than gdp
check_model(fit) #fast check, we can notice collinearity between gdp and potential

ggplot(df) +
  aes(x = time, y = rate) +
  geom_point(aes(y = rate, color = "Interest Rate"), size = 2) +
  geom_point(aes(y = fitted(fit), color = "Fitted Value"), size = 2) +
  scale_color_manual(name = "Legend", values = c("Interest Rate" = "#012622", "Fitted Value" = "#B5446E")) + 
  labs(x = "Year", y = "Interest Rate") + 
  geom_smooth(method = 'lm', color = '#B5446E')

#alternative formulation: using logarithms for gdp and potential output
logfit <- lm(data =df, rate ~ target + actual + log_gdp + log_potential)
summary(logfit)
#we lose the sign match with the Taylor rule but R^2 is higher --> model more precise

ggplot(df) +
  aes(x = time, y = rate) +
  geom_point(aes(y = rate, color = "Interest Rate"), size = 2) +
  geom_point(aes(y = fitted(fit), color = "Fitted Value (Logarithms)"), size = 2) +
  scale_color_manual(name = "Legend", values = c("Interest Rate" = "#012622", "Fitted Value (Logarithms)" = "#EB8258")) + 
  labs(x = "Year", y = "Interest Rate") +
  geom_smooth(method = 'lm', color = '#EB8258')

###TESTS###
check_model(logfit)
# We can notice collinearity between log_gdp & log_potential (understandable, also their original values were collinear)

#Testing linearity of parameter --> Ramsey RESET test
logfit_RESET <- resettest(logfit, type = "fitted", power = c(2:3))
logfit_RESET
# Fail to reject null hypothesis --> model NOT correctly specified

#Homoscedasticity test --> Breusch-Pagan test
logfit_BP <- bptest(logfit)
logfit_BP
#Fail to reject null --> No sufficient evidence of heteroscedasticity in the model

#Hoscedasticity test --> Goldfelt-Quandt
logfit_GQ <- gqtest(logfit, order.by = ~ target + actual + log_gdp + log_potential, data = df, fraction = 17)
logfit_GQ
#We reject the null hypothesis --> heteroscedasticity is present

#Hoscedasticity test --> White test
logfit_white <- bptest(logfit, ~ target + actual + log_gdp + log_potential, data = df)
logfit_white
#Fail to reject the null hypothesis --> Homoscedastic

#Serial correlation test --> Durbin-Watson test
logfit_DW <- durbinWatsonTest(logfit)
logfit_DW
# Fail to reject --> No serial correlation

#Serial correlation test --> LM test
logfit_LM <- bgtest(logfit, order = c(2), order.by = ~target + actual + log_gdp + log_potential,
       data = df)
logfit_LM
#Fail to reject H0 --> No serial correlation

#Normality in error terms --> Jarque-Bera test
logfit_JB <- JarqueBera.test(logfit$residuals)
logfit_JB
#Reject H0 in all 3 tests --> Errors are not normally distributed

#All test done, only issue could be heteroscedasticity
#Possible causes:
# - low significance of log_gdp
# - collinearity of log gdp and log pot (più probabile)




### Extended model ###
# Performing feature selection on extended model
# Adding unemployement rate and rate of exchange NOR/USD

fit_ext <- lm(data = df, formula = rate ~ actual + log_gdp + log_potential + gdp + potential + unemployement_rate + exchange_usd)
summary(fit_ext)

check_model(fit_ext) # Quick analysis of the model to understand issues (only high correlation between variables and their log)

# Performing best subset selection --> Small number of variables, it's a feasible option
bestsub <- regsubsets(rate ~ ., data = df, force.out = "time", method = "exhaustive")
summ_bestsub <- summary(bestsub)
plot(c(1:8), summ_bestsub$adjr2) #we can observe maximum value at 5 feature
coef(bestsub, 5) #This gets the value and the features

# Double-check: performing LASSO selection
# Creating vectors to perform the LASSO 
y <- df$rate
x <- data.matrix(df[, c('target', 'actual', 'gdp', 'potential', 'log_gdp', 'log_potential', 'unemployement_rate', 'exchange_usd')])

#perform k-fold cross-validation to find optimal lambda value
lasso_model <- cv.glmnet(x, y, alpha = 1)

#find optimal lambda value that minimizes test MSE
lasso_lambda <- lasso_model$lambda.min
lasso_lambda

#produce plot of test MSE by lambda value
plot(lasso_model) 

#find coefficients of best model
best_lasso <- glmnet(x, y, alpha = 1, lambda = lasso_lambda)
coef(best_lasso)

#calculating R^2: use fitted best model to make predictions
y_lasso <- predict(lasso_model, s = lasso_lambda, newx = x)

#find SST and SSE
lasso_sst <- sum((y - mean(y))^2)
lasso_sse <- sum((y_lasso - y)^2)

#find R-Squared
lasso_rsq <- 1 - lasso_sse/lasso_sst
lasso_rsq


#lasso regression suggests a different model, with no discarded coefficient

# Double-check: performing ridge regression
ridge_model <- cv.glmnet(x, y, alpha = 0)

ridge_lambda <- ridge_model$lambda.min
ridge_lambda

plot(ridge_model)

best_ridge <- glmnet(x, y, alpha = 0, lambda = ridge_lambda)
coef(best_ridge)
# Here gdp and potential gdp can be discarded

#calculating R^2
y_ridge <- predict(ridge_model, s = ridge_lambda, newx = x)

ridge_sst <- sum((y - mean(y))^2)
ridge_sse <- sum((y_ridge - y)^2)

ridge_rsq <- 1 - ridge_sse/ridge_sst
ridge_rsq

