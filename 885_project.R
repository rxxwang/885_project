library(survival)
library(cmprsk)
library(ggplot2)
library(gridExtra)
library(dplyr)
library(tidyr)
library(splines)
library(mgcv)
library(gtsummary)
library(tidyverse)
library(kableExtra)

# descriptive table
summary1 = jasa %>% select(fustat, surgery, age, transplant) %>%
  tbl_summary(by = transplant, statistic = list(all_continuous() ~ "{mean} ({sd})")) %>%
  as.data.frame()
jasa %>% select(fustat, surgery, age) %>%
  tbl_summary(statistic = list(all_continuous() ~ "{mean} ({sd})"))  %>%
  as.data.frame() %>% right_join(summary1) %>%
  kable(format = "latex") %>% print()

# Histogram of age
hist <- jasa %>% mutate(Transplant = case_when(transplant == 0 ~ "No", transplant == 1 ~ "Yes")) %>%
  ggplot(aes(x = age)) + 
  geom_histogram(aes(y = ..density.., fill = Transplant, group = Transplant), alpha = 0.5, position = "identity") +
  labs(x = "Age", y = "Probability") + theme_bw()+ 
  theme(plot.title = element_text(hjust = 0.5))
ggsave(hist, file = "Histogram.png", width = 6, height = 4)

# CIF plot
cif_data = jasa %>% mutate(Transplant = factor(case_when(transplant == 0 ~ "Non-Transplant", 
                                                         transplant == 1 ~ "Transplant"),
                                               levels = c("Transplant", "Non-Transplant")),
                           futime2 = futime/365,
                           status = case_when(fustat == 0 ~ "Dead", fustat == 1 ~ "Survival"))
cif <- cuminc(ftime = cif_data$futime2, fstatus = cif_data$status, group = cif_data$Transplant, cencode = "Censored")

cif_tidy <- do.call(rbind,lapply(1:2, function(i) {
  data.frame(
    time = cif[[i]]$time,
    cif = cif[[i]]$est,
    group = names(cif)[i]
  )
})
) %>%
  separate(group, into = c("Transplant", "Event"), sep = " ")

cif_plot <- ggplot(cif_tidy, aes(x = time, y = cif, color = Transplant)) +
  geom_line() +
  labs(
    x = "Time (Years)",
    y = "Cumulative Incidence",
    color = "Heart Transplantation"
  ) + ylim(0, 0.4) + theme_bw()
ggsave(cif_plot, file = "cif_plot.png", width = 6, height = 4)

# save the data to import in python
data <- jasa %>% select(age, transplant, fustat)
write.csv(data, file = "heart.csv", row.names = F)

# Logistic Regression
data$Transplant = factor(ifelse(data$transplant == 0, "N", "Y"), levels = c("Y", "N"))
logs <- glm(fustat ~ age + Transplant:age, family = binomial, data = data)
age_grid = seq(min(data$age), max(data$age), by = 0.1)
grid = data.frame(
  age = age_grid,
  Transplant = factor(c(rep("Y", length(age_grid)), rep("N", length(age_grid))), levels = c("Y", "N"))
)
y_logs = predict(logs, newdata = grid, type = "link", se.fit = T)

logs_pred <- data.frame(
  age = grid$age,
  transplant = grid$Transplant,
  y_upper = plogis(y_logs$fit + 1.96 * y_logs$se.fit),
  y_lower = plogis(y_logs$fit - 1.96 * y_logs$se.fit),
  y_pred = plogis(y_logs$fit)
)

ggplot() + 
  geom_line(data = logs_pred, aes(x = age, y = y_pred, color = transplant))+
  geom_ribbon(data = logs_pred, aes(x = age, ymin = y_lower, ymax = y_upper, fill = transplant), alpha = 0.2)+
  geom_point(data = data, aes(x = age, y = fustat, color = Transplant)) + 
  labs(x = "Age", y = "Survival", color = "Transplant", fill = "Transplant")+
  theme_bw()
ggsave("logistic.png", width = 8, height = 3)

# GAM
data$Transplant = factor(ifelse(data$transplant == 0, "N", "Y"), levels = c("Y", "N"))

aic = rep(0,6)
for(i in 3:6){
  gam.model <- gam(fustat ~ s(age, by = Transplant, k = i), family = binomial, data = data)
  aic[i] = gam.model$aic
}
print(aic[3:6])

gam.model <- gam(fustat ~ s(age, by = Transplant, k = 5), family = binomial, data = data)
age_grid = seq(min(data$age), max(data$age), by = 0.1)
grid = data.frame(
  age = age_grid,
  Transplant = factor(c(rep("Y", length(age_grid)), rep("N", length(age_grid))), levels = c("Y", "N"))
)
y_gam = predict(gam.model, newdata = grid, type = "link", se.fit = T)

gam_pred <- data.frame(
  age = grid$age,
  transplant = grid$Transplant,
  y_upper = plogis(y_gam$fit + 1.96 * y_gam$se.fit),
  y_lower = plogis(y_gam$fit - 1.96 * y_gam$se.fit),
  y_pred = plogis(y_gam$fit)
)

ggplot() + 
  geom_line(data = gam_pred, aes(x = age, y = y_pred, color = transplant))+
  geom_ribbon(data = gam_pred, aes(x = age, ymin = y_lower, ymax = y_upper, fill = transplant), alpha = 0.2)+
  geom_point(data = data, aes(x = age, y = fustat, color = Transplant)) + 
  labs(x = "Age", y = "Survival", color = "Transplant", fill = "Transplant")+
  theme_bw()
ggsave("gam.png", width = 8, height = 3)

# Transform data to fit glmm
n = length(data$transplant)
x = data$transplant
z = round(data$age, digits = 0)
y = data$fustat

z0 = sort(unique(z))
N = matrix(0, ncol = length(z0), nrow = n)
group1 = rep(0, n)
for (i in 1:n) {
  for (j in 1:length(z0)) {
    if(z[i] == z0[j]){
      N[i, j] = 1
      group1[i] = j
    }
  }
}
m = length(z0)

R = matrix(0, ncol = m - 2, nrow = m - 2)
h = z0[-1] - z0[-m]
for (i in 1:(m - 2)) {
  R[i, i] = 1 / 3 * (h[i] + h[i + 1])
  if (i < m - 2) {
    R[i, i + 1] = 1 / 6 * h[i + 1]
    R[i + 1, i] = 1 / 6 * h[i + 1]
  }
}
Q = matrix(0, ncol = m - 2, nrow = m)
for (i in 1:(m - 2)) {
  Q[i, i] = 1 / h[i]
  Q[i + 1, i] = -1 / h[i] - 1 / h[i + 1]
  Q[i + 2, i] = 1 / h[i + 1]
}
L = Q %*% t(chol(solve(R)))
B = L %*% solve(t(L) %*% L)

glmer_data = data.frame(
  x = x,
  y = y,
  z = z,
  group1 = factor(group1),
  xz = x * z
)
NB = N %*% B
xNB = x * N %*% B

# Import GLMM parameters form python (by hand)
random1 = c(0.014535689726471901, 0.01827911287546158, 0.04250914976000786, 0.04585462063550949, 0.022177614271640778, 0.037991318851709366, 0.03060781955718994, 0.015097898431122303, 0.0066600823774933815, 0.005662161391228437, -0.00200834684073925, -0.0051585352048277855, -0.012192809954285622, -0.016992434859275818, -0.018562819808721542, -0.0010996811324730515, 0.004063690081238747, 0.0023382697254419327, 0.014094613492488861, 0.013980633579194546, 0.012011805549263954, 0.006078487262129784, 0.007177082356065512, 0.005466470494866371, -0.0017936171498149633, 0.0014696376165375113, 0.0003766840964090079, -0.00012901093577966094, -0.0011572587536647916, -0.0002545220195315778, -0.0010971983429044485)
random2 = c(0.012201734818518162, 0.012431493028998375, 0.02506319433450699, 0.021552419289946556, 0.014366020448505878, 0.014623437076807022, 0.013681814074516296, 0.006327250972390175, 0.0019895746372640133, 0.0007521318621002138, -0.005593017674982548, -0.007034580688923597, -0.010561737231910229, -0.015216526575386524, -0.005424955394119024, 0.001302984543144703, 0.005245485808700323, 0.010253772139549255, 0.013131825253367424, 0.010451552458107471, 0.006816343404352665, 0.004251660313457251, 0.0049791475757956505, 0.0052757118828594685, 0.0035022988449782133, 0.0014335352461785078, 0.0004682332801166922, -2.6174386221100576e-05, -0.00014417014608625323, 0.0007372093386948109, -0.0003009156498592347)
coef = c(-0.4035, -0.2368, 0.0646, -0.0289)

# Plot Smoothing Spline
theta_0 = as.vector(coef[1] * rep(1, n) + z * coef[3] + N %*% B %*% random1)
theta_1 = as.vector(coef[2] * rep(1, n) + z * coef[4] + N %*% B %*% random2)

glmm_pred = data.frame(
  age = rep(z,2),
  transplant = c(rep("Y",n),rep("N",n)),
  y_pred = c(plogis(theta_0 + theta_1),plogis(theta_0))
)

ggplot() + 
  geom_line(data = glmm_pred, aes(x = age, y = y_pred, color = transplant))+
  geom_point(data = data, aes(x = age, y = fustat, color = Transplant)) + 
  labs(x = "Age", y = "Survival", color = "Transplant", fill = "Transplant")+
  theme_bw()
ggsave("glmm.png", width = 8, height = 3)