library(RDS)
args <- commandArgs(trailingOnly = TRUE)
input_csv  <- args[1]
output_csv <- args[2]
N          <- as.numeric(args[3])

data <- read.csv(input_csv, stringsAsFactors = FALSE)

# Ensure seeds have unique recruiter ID
data$recruiter[data$recruiter == "seed"] <- paste0("seed_", seq_len(sum(data$recruiter == "seed")))

# --- FIX HERE! Use 'network.size' not 'network_size' ---
rds <- as.rds.data.frame(data, id = "id", recruiter.id = "recruiter", network.size = "network.size")

# Compute Gile's SS weights
weights <- compute.weights(rds, weight.type = "Gile's SS", N = N)

# Fallback if weight names are missing
if (is.null(names(weights))) {
  names(weights) <- data$id
}

# Combine and write
result <- data.frame(id = names(weights), weight = as.vector(weights))
colnames(result) <- c("id", "weight")
write.csv(result, file = output_csv, row.names = FALSE)




