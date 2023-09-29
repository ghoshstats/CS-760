#################### Q2 :: (2) #####################################################
data <- data.frame(x1 = c(0, 1, 0, 1), x2 = c(0, 0, 1, 1), y = c(0, 1, 1, 0))

# Plot the points
plot(data$x1, data$x2, col = data$y + 1, pch = 19, xlim = c(-0.5, 1.5), ylim = c(-0.5, 1.5), xlab = "x1", ylab = "x2")
abline(v=0.5, col="blue", lty=2)
abline(h=0.5, col="red", lty=2)

# Add a legend
legend("topright", legend = c("Class 0", "Class 1"), fill = 1:2)

#################### Q2 :: (3) #####################################################
# Modified function to list all candidate splits and their information gain ratio or mutual information
find_candidate_splits <- function(data) {
  candidate_splits_info <- data.frame()
  parent_entropy <- calculate_entropy(data)
  
  for (j in c('x1', 'x2')) {
    values <- unique(data[[j]])
    
    for (c in values) {
      left <- filter(data, get(j) < c)
      right <- filter(data, get(j) >= c)
      
      if (nrow(left) == 0 || nrow(right) == 0) next
      
      entropy_left <- calculate_entropy(left)
      entropy_right <- calculate_entropy(right)
      
      total <- nrow(data)
      gain <- parent_entropy - (nrow(left) / total * entropy_left + nrow(right) / total * entropy_right)
      
      info <- split_information(left, right)
      if (info == 0) next
      
      if (entropy_left == 0 && entropy_right == 0) {
        candidate_splits_info <- rbind(candidate_splits_info, data.frame(feature=j, threshold=c, type='Mutual Information', value=gain))
      } else {
        gain_ratio <- gain / info
        candidate_splits_info <- rbind(candidate_splits_info, data.frame(feature=j, threshold=c, type='Gain Ratio', value=gain_ratio))
      }
    }
  }
  
  candidate_splits_info
}

# Find and list all candidate splits and their information gain ratio or mutual information
#Druns <- read.table("~/Downloads/Homework 2 data/Druns.txt", quote="\"", comment.char="")

colnames(Druns) <- c("x1", "x2", "label")
candidate_splits_info <- find_candidate_splits(Druns)
print(candidate_splits_info)

##################### Q2 :: (5) ######################################################

#D2 <- read.table("~/Downloads/Homework 2 data/D2.txt", quote="\"", comment.char="")
colnames(D2) = c("x1", "x2", "label")

# Define Functions
calculate_entropy <- function(data) {
  if (nrow(data) == 0) return(0)
  label_counts <- table(data$label)
  probs <- label_counts / sum(label_counts)
  -sum(probs * log2(probs))
}

split_information <- function(left, right) {
  total <- nrow(left) + nrow(right)
  if (total == 0) return(0)
  
  p_left <- nrow(left) / total
  p_right <- nrow(right) / total
  
  info <- 0
  if (p_left > 0) info <- info - p_left * log2(p_left)
  if (p_right > 0) info <- info - p_right * log2(p_right)
  
  info
}

find_best_split <- function(data) {
  best_gain_ratio <- 0
  best_split <- NULL
  parent_entropy <- calculate_entropy(data)
  
  for (j in c("x1", "x2")) {
    values <- unique(data[[j]])
    
    for (c in values) {
      left <- filter(data, get(j) < c)
      right <- filter(data, get(j) >= c)
      
      if (nrow(left) == 0 || nrow(right) == 0) next
      
      entropy_left <- calculate_entropy(left)
      entropy_right <- calculate_entropy(right)
      
      total <- nrow(data)
      gain <- parent_entropy - (nrow(left) / total * entropy_left + nrow(right) / total * entropy_right)
      
      info <- split_information(left, right)
      if (info == 0) next
      
      gain_ratio <- gain / info
      if (gain_ratio > best_gain_ratio) {
        best_gain_ratio <- gain_ratio
        best_split <- list(feature=j, threshold=c, gain_ratio=gain_ratio)
      }
    }
  }
  
  best_split
}

build_tree <- function(data) {
  if (nrow(data) == 0) return(list(label=1))
  if (calculate_entropy(data) == 0) return(list(label=as.integer(unique(data$label))))
  
  split <- find_best_split(data)
  if (is.null(split)) return(list(label=as.integer(ifelse(length(unique(data$label)) > 1, 1, unique(data$label)))))
  
  left_data <- filter(data, get(split$feature) < split$threshold)
  right_data <- filter(data, get(split$feature) >= split$threshold)
  
  left_branch <- build_tree(left_data)
  right_branch <- build_tree(right_data)
  
  list(feature=split$feature, threshold=split$threshold, left=left_branch, right=right_branch)
}

print_tree <- function(tree, indent="") {
  if (!is.null(tree$label)) {
    cat(indent, "Label: ", tree$label, "\n")
    return()
  }
  
  cat(indent, tree$feature, " >= ", tree$threshold, "\n", sep="")
  cat(indent, "Then:\n")
  print_tree(tree$left, paste0(indent, "  "))
  cat(indent, "Else:\n")
  print_tree(tree$right, paste0(indent, "  "))
}

# Build and Print Tree
tree <- build_tree(D2)
print_tree(tree)

#################### Q2:: (6)##################################################
library(RColorBrewer)

#D1 <- read.table("~/Downloads/Homework 2 data/D1.txt", quote="\"", comment.char="")
#D2 <- read.table("~/Downloads/Homework 2 data/D2.txt", quote="\"", comment.char="")

# Load the data
colnames(D1) = c('x1', 'x2', 'label')
colnames(D2) = c('x1', 'x2', 'label')

# Function to visualize the scatter plots and decision boundaries
plot_data_and_boundaries <- function(data, tree, title) {
  
  # Assuming predict_tree is a function you have that predicts the label for given features.
  get_predictions <- function(data, tree) {
    predictions <- numeric(nrow(data))
    for(i in 1:nrow(data)) {
      predictions[i] <- predict_tree(tree, data[i,])
    }
    predictions
  }
  
  # Define a grid over the feature space and get predictions for each point in the grid
  grid <- expand.grid(x1 = seq(min(data$x1) - 0.1, max(data$x1) + 0.1, length.out = 100),
                      x2 = seq(min(data$x2) - 0.1, max(data$x2) + 0.1, length.out = 100))
  grid$label <- get_predictions(grid, tree)
  
  # Plot the decision boundary and overlay the scatter plot of the original data
  plot <- ggplot(data = grid, aes(x = x1, y = x2)) +
    geom_tile(aes(fill = as.factor(label)), alpha = 0.3) +
    scale_fill_manual(values = brewer.pal(3, "Set1"), name = "Class Label", labels = c("Class 0", "Class 1")) +
    geom_point(data = data, aes(color = as.factor(label))) +
    scale_color_manual(values = brewer.pal(3, "Set1"), name = "Class Label", labels = c("Class 0", "Class 1")) +
    labs(title = title) +
    theme_minimal() +
    theme(legend.position = "bottom", plot.title = element_text(hjust = 0.5))
  
  print(plot)
  
  
}

tree_d1 <- build_tree(D1)
tree_d2 <- build_tree(D2)

# Visualize the scatter plots and decision boundaries
plot_data_and_boundaries(D1, tree_d1, 'D1.txt: Scatter Plot and Decision Boundary')
plot_data_and_boundaries(D2, tree_d2, 'D2.txt: Scatter Plot and Decision Boundary')

