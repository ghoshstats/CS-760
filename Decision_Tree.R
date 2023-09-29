################## Q1 :: Implementing a decision tree from scratch #####################
########################################################################################
library(dplyr)

# Define a function to calculate the entropy of a dataset
calculate_entropy <- function(data) {
  if (nrow(data) == 0) return(0)
  label_counts <- table(data$label)
  probs <- label_counts / sum(label_counts)
  -sum(probs * log2(probs))
}

# Define a function to calculate split information
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

# Finding the best split based on information gain ratio
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

# Define a recursive function to build the decision tree
build_tree <- function(data) {
  if (nrow(data) == 0) return(list(label=1))
  if (entropy(data) == 0) return(list(label=as.integer(names(which.max(table(data$label))))))
  
  split <- find_best_split(data)
  if (is.null(split)) return(list(label=as.integer(ifelse(length(unique(data$label)) > 1, 1, unique(data$label)))))
  
  left_data <- filter(data, get(split$feature) < split$threshold)
  right_data <- filter(data, get(split$feature) >= split$threshold)
  
  left_branch <- build_tree(left_data)
  right_branch <- build_tree(right_data)
  
  list(feature=split$feature, threshold=split$threshold, left=left_branch, right=right_branch)
}

# Define a function to make predictions using the decision tree
predict_tree <- function(tree, point) {
  if (!is.null(tree$label)) return(tree$label)
  if (point[tree$feature] < tree$threshold) return(predict_tree(tree$left, point))
  return(predict_tree(tree$right, point))
}

# # Load the data
# data <- data.frame(x1=c(1.0, 2.0, 3.0, 4.0), x2=c(2.0, 3.0, 4.0, 5.0), label=c(0, 1, 0, 1))
# 
# # Build the decision tree
# tree <- build_tree(data)
# 
# # Make predictions
# predictions <- sapply(1:nrow(data), function(i) predict_tree(tree, data[i,]))
# print(predictions)

