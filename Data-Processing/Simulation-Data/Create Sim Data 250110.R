library(MASS)
library(ggplot2)
library(truncnorm)

### TEST : truncnorm ### (block comment shortcut : Ctrl Shift C)
# AGE <- rtruncnorm(5000, a = 18, b = 90, mean = 55, sd = 20)
# summary(AGE)

# Function to simulate vancomycin data # dur : infusion duration (default : 1hr)
simulate_vanco <- function(n_subj, age_mean, age_sd, wt_mean, wt_sd, cr_mean, cr_sd, 
                           dose = 1000, dosing_interval = 12, total_doses = 28, dur = 1) {
  # Randomly assign demographics
  SEX <- rbinom(n_subj, 1, 0.5)  # 0 = male, 1 = female
  AGE <- rtruncnorm(n_subj, a = 18, b = 90, mean = age_mean, sd = age_sd) # Truncated Distribution
  WT <- rtruncnorm(n_subj, a = 40, b = 150, mean = wt_mean, sd = wt_sd)
  Cr <- rtruncnorm(n_subj, a = 0.4, b = 5, mean = cr_mean, sd = cr_sd)
  
  # Dosing schedule
  dosing_times <- seq(0, by = dosing_interval, length.out = total_doses)
  doses <- rep(dose, length(dosing_times))
  
  # Time grid for ground truth
  time <- seq(0, 372, by = 0.1)
  
  # Variability: Sampling from MVN
  eta_cov <- matrix(c(0.120, 0, 0, 
                      0, 0.149, 0, 
                      0, 0, 0.416), nrow = 3)
  eta <- mvrnorm(n_subj, mu = c(0, 0, 0), Sigma = eta_cov)
  
  # PK Parameter Calculation Function
  calc_PK <- function(eta_i, age, wt, cr, is_female) {
    CCR <- ((140 - age) * wt * ifelse(is_female, 0.85, 1)) / (72 * cr)
    V1 <- 33.1 * exp(eta_i[1])
    V2 <- 48.3
    CL <- 3.96 * (CCR / 100) * exp(eta_i[2])
    Q  <- 6.99 * exp(eta_i[3])
    k10 <- CL / V1
    k12 <- Q / V1
    k21 <- Q / V2
    lambda1 <- (k10 + k12 + k21 + sqrt((k10 + k12 + k21)^2 - 4 * k10 * k21)) / 2
    lambda2 <- (k10 + k12 + k21) - lambda1
    list(V1 = V1, V2 = V2, CL = CL, Q = Q, k10 = k10, k12 = k12, k21 = k21, 
         lambda1 = lambda1, lambda2 = lambda2)
  }
  
  # Precompute PK Parameters
  PK_params <- lapply(1:n_subj, function(i) calc_PK(eta[i, ], AGE[i], WT[i], Cr[i], SEX[i]))
  
  # Save PK parameters to a data frame
  pk_params_df <- do.call(rbind, lapply(1:n_subj, function(i) {
    pk <- PK_params[[i]]
    data.frame(
      ID = i,
      SEX = SEX[i],
      AGE = AGE[i],
      WT = WT[i],
      Cr = Cr[i],
      V1 = pk$V1,
      V2 = pk$V2,
      CL = pk$CL,
      Q = pk$Q,
      k10 = pk$k10,
      k12 = pk$k12,
      k21 = pk$k21,
      lambda1 = pk$lambda1,
      lambda2 = pk$lambda2
    )
  }))
  
  # Simulate Concentration-Time Profile
  simulate_concentration <- function(pk, times, doses, dosing_times, dur) {
    C1 <- (pk$lambda1 - pk$k21) / (pk$V1 * (pk$lambda1 - pk$lambda2))
    C2 <- (pk$k21 - pk$lambda2) / (pk$V1 * (pk$lambda1 - pk$lambda2))
    
    sapply(times, function(t) {
      sum(sapply(seq_along(dosing_times), function(i) {
        d <- dosing_times[i]
        d_amt <- doses[i]
        Rate <- d_amt / dur
        if (t >= d && t <= (d + dur)) { 
          # During infusion
          term1 <- (Rate / pk$lambda1) * C1 * (1 - exp(-pk$lambda1 * (t - d)))
          term2 <- (Rate / pk$lambda2) * C2 * (1 - exp(-pk$lambda2 * (t - d)))
          term1 + term2
        } else if (t > (d + dur)) {
          # After infusion
          term1 <- (Rate / pk$lambda1) * C1 * (1 - exp(-pk$lambda1 * dur)) * exp(-pk$lambda1 * (t - (d + dur)))
          term2 <- (Rate / pk$lambda2) * C2 * (1 - exp(-pk$lambda2 * dur)) * exp(-pk$lambda2 * (t - (d + dur)))
          term1 + term2
        } else {
          # Before infusion
          0
        }
      }))
    })
  }
  
  # Generate Ground Truth Data
  ground_truth <- do.call(rbind, lapply(1:n_subj, function(i) {
    pk <- PK_params[[i]]
    conc <- simulate_concentration(pk, time, doses, dosing_times, dur)
    TAD <- sapply(time, function(t) max(dosing_times[dosing_times <= t], na.rm = TRUE))
    data.frame(ID = i, TIME = time, TAD = time - TAD, AMT = NA, DV = conc,
               SEX = SEX[i], AGE = AGE[i], WT = WT[i], Cr = Cr[i])
  }))
  
  # Add dosing information
  dose_data <- do.call(rbind, lapply(1:n_subj, function(i) {
    data.frame(ID = i, TIME = dosing_times, TAD = 0, AMT = doses, DV = NA,
               SEX = SEX[i], AGE = AGE[i], WT = WT[i], Cr = Cr[i])
  }))
  
  ground_truth <- rbind(ground_truth, dose_data)
  ground_truth <- ground_truth[order(ground_truth$ID, ground_truth$TIME), ]
  
  # Generate Observation Data
  obs_times <- c(72, 144, 216, 288)
  observation <- subset(ground_truth, TIME %in% obs_times | !is.na(AMT))
  
  # Output
  list(ground_truth = ground_truth, observation = observation, pk_params = pk_params_df)
}

# Simulate data
result <- simulate_vanco(
  n_subj = 3000, 
  age_mean = 55, age_sd = 20,
  wt_mean = 70, wt_sd = 15,
  cr_mean = 1.0, cr_sd = 0.3, 
  dur = 1  # Infusion duration in hours
)

ground_truth <- result$ground_truth
observation <- result$observation
pk_params <- result$pk_params

# Save to CSV
setwd("C:/Users/USER/Documents/R_project/RxODE/Data Generated (Sung Pil Han)")
write.csv(ground_truth, "ground_truth_250110.csv", row.names = FALSE)
write.csv(observation, "observation_250110.csv", row.names = FALSE)
write.csv(pk_params, "pk_parameters_250110.csv", row.names = FALSE)
