# Load packages
library(tidyverse)
library(ggplot2)
library(caret)
library(tidytext)
library(kableExtra)
library(rpart)
library(keras)
library(tfhub)

# Load data and Pre-Process
load("sona.RData")

# Remove date appearing at the beginning of speech
x <- sona$speech
y <- sub('^\\w*\\s*\\w*\\s*\\w*\\s*', '', x[1:34])
sona$speech[1:34] <- y

# Format 35th speech 
z <- sub("^[A-Za-z]+, \\d{1,2} [A-Za-z]+ \\d{4}  ", "", x[35])
sona$speech[35] <- z

# Format 36th speech
a <- sub("\\d{1,2} [A-Za-z]+ \\d{4}", "", x[36])
sona$speech[36] <- a

# Remove all non-alphanumeric characters
sona$speech <- str_replace_all(sona$speech, "[^[:alnum:]]", " ")

# Convert to appropriate format
sona$president_13 <- as.factor(sona$president_13)
sona$year <- as.numeric(sona$year)
sona$date <- as.Date(sona$date,format = "%d-%m-%Y")
tidy_sona <- sona %>% unnest_tokens(word, speech, token = "words", to_lower = T)

#### EDA ####
# Plot speeches by president
speech_count_per_president <- sona %>%
  group_by(president_13) %>%
  summarize(speech_count = n())

# Print the speech count per president
print(speech_count_per_president)

# Create a bar plot for speeches per president
ggplot(speech_count_per_president, aes(x = president_13, y = speech_count, fill = president_13)) +
  geom_bar(stat = "identity") +
  labs(title = "Number of Speeches per President",
       x = "President",
       y = "Number of Speeches") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "none") +
  scale_fill_discrete(name = "President")

# Tokenize and count words per president
word_count_per_president <- sona %>%
  mutate(word_count = lengths(strsplit(speech, " "))) %>%
  group_by(president_13) %>%
  summarize(total_words = sum(word_count))

# Print the word count per president
print(word_count_per_president)

# Create a bar plot for words per president
ggplot(word_count_per_president, aes(x = president_13, y = total_words, fill = president_13)) +
  geom_bar(stat = "identity") +
  labs(title = "Number of Words per President",
       x = "President",
       y = "Total Words") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "none") +
  scale_fill_discrete(name = "President")

# Plot 20 most used words in speeches (excluding stop words)
sona %>% 
  unnest_tokens(word, speech, token = 'words') %>%
  count(word, sort=TRUE) %>% 
  filter(!word %in% stop_words$word) %>% 
  filter(rank(desc(n)) <= 20) %>% 
  ggplot(aes(x = reorder(word, n), y = n)) + geom_col() + coord_flip() + xlab("")


#### Bag-of-words ####
word_bag <- tidy_sona %>%
  filter(!word %in% stop_words$word) %>% 
  group_by(word) %>% 
  count() %>%
  ungroup() %>% 
  top_n(200, wt = n) %>%
  select(-n)

sona_tdf <- tidy_sona %>%
  inner_join(word_bag) %>%
  group_by(filename, president_13, word) %>%
  count() %>%  
  group_by(filename) %>%
  mutate(total = sum(n)) %>%
  ungroup()

bag_of_words <- sona_tdf %>% 
  select(filename, word, n) %>% 
  pivot_wider(names_from = word, values_from = n, values_fill = 0) %>%
  left_join(sona %>% select(filename, president_13)) %>%
  select(filename, president_13, everything())

# Building a classifier
set.seed(321)
sample_index <- createDataPartition(bag_of_words$president_13, p = 0.7, list = FALSE)
training_ids <- bag_of_words[sample_index, ] %>%
  select(filename)

#### Decision Tree ####
training_sona <- bag_of_words %>%
  right_join(training_ids, by = 'filename') %>% 
  select(-filename)

test_sona <- bag_of_words %>%
  anti_join(training_ids, by = 'filename') %>% 
  select(-filename)

fit <- rpart(president_13 ~ ., data = training_sona, method = 'class')
predictions <- predict(fit, test_sona, type = 'class')
pred_test <- table(test_sona$president_13, predictions)
round(sum(diag(pred_test))/sum(pred_test), 3) 

#### Neural Network ####
training_rows <- sample_index

train <- list()
test <- list()
train$x <- as.matrix(bag_of_words[training_rows,-c(1,2)])
test$x <-  as.matrix(bag_of_words[-training_rows,-c(1,2)])

train$y <- to_categorical(as.integer(bag_of_words$president_13[training_rows]) - 1)
test$y <-  to_categorical(as.integer(bag_of_words$president_13[-training_rows]) - 1)

tensorflow::set_random_seed(4)
model <- keras_model_sequential() %>%
  layer_dense(units = 32, input_shape = dim(train$x)[2], activation = "relu") %>% 
  layer_dropout(rate=0.2) %>% 
  layer_dense(units = 6, activation = "softmax")

model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = "adam",
  metrics = "accuracy"
)

history <- model %>% fit(train$x, train$y, epochs = 65, 
                         batch_size = 20,  validation_split = 0.1, verbose = 0) 
plot(history)

results <- model %>% evaluate(test$x, test$y, batch_size = 20, verbose = 2)
results

#### TF-IDF ####
sona_tdf <- tidy_sona %>%
  inner_join(word_bag) %>%
  group_by(filename, president_13, word) %>%
  count() %>%  
  group_by(filename) %>%
  mutate(total = sum(n)) %>%
  ungroup()

ndocs <- length(unique(sona_tdf$filename))

# Calculate idf
idf <- sona_tdf %>% 
  group_by(word) %>% 
  summarize(docs_with_word = n()) %>% 
  ungroup() %>%
  mutate(idf = log(ndocs / docs_with_word)) %>% arrange(desc(idf))

# Calculate tf-idf
sona_tdf <- sona_tdf %>% 
  left_join(idf, by = 'word') %>% 
  mutate(tf = n/total, tf_idf = tf * idf)

# ***Alternatively
#sona_tdf <- sona_tdf %>% bind_tf_idf(word, filename, n)
  
# Reshape dataset
tfidf <- sona_tdf %>% 
  select(filename, word, tf_idf) %>%  # note the change, using tf-idf
  pivot_wider(names_from = word, values_from = tf_idf, values_fill = 0) %>%  
  left_join(sona %>% select(filename, president_13))

# Building a classifier
set.seed(321)
sample_index <- createDataPartition(tfidf$president_13, p = 0.7, list = FALSE)
training_ids <- tfidf[sample_index, ] %>%
  select(filename)

#### Decision Tree ####
training_sona <- tfidf %>% 
  right_join(training_ids, by = 'filename') %>%
  select(-filename)

test_sona <- tfidf %>% 
  anti_join(training_ids, by = 'filename') %>%
  select(-filename)

fit <- rpart(president_13 ~ ., data = training_sona, method = 'class')
predictions <- predict(fit, test_sona, type = 'class')
pred_test <- table(test_sona$president_13, predictions)
round(sum(diag(pred_test))/sum(pred_test), 3) 

#### Neural Network ####
set.seed(321)
sample_index <- createDataPartition(tfidf$president_13, p = 0.7, list = FALSE)
training_ids <- tfidf[sample_index, ] %>%
  select(filename)
training_rows <- sample_index

train <- list()
test <- list()
train$x <- as.matrix(tfidf[training_rows,-c(1,202)])
test$x <-  as.matrix(tfidf[-training_rows,-c(1,202)])

train$y <- to_categorical(as.integer(tfidf$president_13[training_rows]) - 1)
test$y <-  to_categorical(as.integer(tfidf$president_13[-training_rows]) - 1)

tensorflow::set_random_seed(4)
model <- keras_model_sequential() %>%
  layer_dense(units = 32, input_shape = dim(train$x)[2], activation = "relu") %>% 
  layer_dropout(rate=0.2) %>% 
  layer_dense(units = 6, activation = "softmax")

model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = "adam",
  metrics = "accuracy"
)

history <- model %>% fit(train$x, train$y, epochs = 65, 
                         batch_size = 20,  validation_split = 0.1, verbose = 0) 
plot(history)

results <- model %>% evaluate(test$x, test$y, batch_size = 20, verbose = 2)
results

#### n-grams ####
# tokenisation
sona_trigrams <- sona %>% 
  unnest_tokens(trigram, speech, token = "ngrams", n = 3)

# separate trigrams
trigrams_separated <- sona_trigrams %>% separate(trigram,
                                             c('word1', 'word2', 'word3'),
                                             sep = " ")
# remove stop words
trigrams_filtered <- trigrams_separated %>%
  filter_at(vars(word1, word2, word3), all_vars(!(. %in% stop_words$word)))

# join trigrams again
trigrams_united <- trigrams_filtered %>%
  unite(trigram, word1, word2, word3, sep = " ")

top_trigrams <- trigrams_united %>%
  group_by(president_13, trigram) %>%
  count() %>%
  top_n(200, wt = n) %>%
  select(-n)

sona_tdf <- trigrams_united %>%
  inner_join(top_trigrams) %>%
  group_by(filename, president_13, trigram) %>%
  count() %>%
  group_by(filename) %>%
  mutate(total = sum(n)) %>%
  ungroup()

bag_of_t <- sona_tdf %>%
  select(filename, trigram, n) %>%
  pivot_wider(names_from = trigram, values_from = n, values_fill = 0) %>%
  left_join(sona %>% select(filename, president_13)) %>%
  select(filename, president_13, everything())

# Building a classifier
set.seed(321)
sample_index <- createDataPartition(bag_of_t$president_13, p = 0.7, list = FALSE)
training_ids <- bag_of_t[sample_index, ] %>% select(filename)

#### Decision Tree ####
training_sona <- bag_of_t %>%
  right_join(training_ids, by = 'filename') %>% 
  select(-filename)

test_sona <- bag_of_t %>%
  anti_join(training_ids, by = 'filename') %>% 
  select(-filename)

fit <- rpart(president_13 ~ ., data = training_sona, method = 'class')
predictions <- predict(fit, test_sona, type = 'class')
pred_test <- table(test_sona$president_13, predictions)
round(sum(diag(pred_test))/sum(pred_test), 3) 

#### Neural Network ####
training_rows <- sample_index

train <- list()
test <- list()
train$x <- as.matrix(bag_of_t[training_rows,-c(1,2)])
test$x <-  as.matrix(bag_of_t[-training_rows,-c(1,2)])

train$y <- to_categorical(as.integer(bag_of_t$president_13[training_rows]) - 1)
test$y <-  to_categorical(as.integer(bag_of_t$president_13[-training_rows]) - 1)

tensorflow::set_random_seed(4)
model <- keras_model_sequential() %>%
  layer_dense(units = 32, input_shape = dim(train$x)[2], activation = "relu") %>% 
  layer_dropout(rate=0.2) %>% 
  layer_dense(units = 6, activation = "softmax")

model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = "adam",
  metrics = "accuracy"
)

history <- model %>% fit(train$x, train$y, epochs = 65, 
                         batch_size = 20,  validation_split = 0.1, verbose = 0) 
plot(history)

results <- model %>% evaluate(test$x, test$y, batch_size = 20, verbose = 2)
results

#### Word embeddings ####
#### CNN ####
max_features <- 6000        # choose max_features most popular words
tokenizer = text_tokenizer(num_words = max_features)

fit_text_tokenizer(tokenizer, sona$speech)
sequences = tokenizer$texts_to_sequences(sona$speech)

y <- to_categorical(as.integer(sona$president_13) - 1)

set.seed(321)
sample_index <- createDataPartition(sona$president_13, p = 0.7, list = FALSE)

training_rows <- sample_index

train <- list()
test <- list()
train$x <- sequences[training_rows]
test$x <-  sequences[-training_rows]

train$y <- y[training_rows,]
test$y <-  y[-training_rows,]

hist(unlist(lapply(sequences, length)), main = "Sequence length after tokenization")

maxlen <- 9000               
x_train <- train$x %>% pad_sequences(maxlen = maxlen)
x_test <- test$x %>% pad_sequences(maxlen = maxlen)

embedding_dims <- 50
tensorflow::set_random_seed(4)
model <- keras_model_sequential() %>% 
  layer_embedding(max_features, output_dim = embedding_dims, input_length = maxlen) %>%
  layer_dropout(0.2) %>%
  layer_conv_1d(filters = 64, kernel_size = 8, activation = "relu") %>%
  layer_max_pooling_1d(pool_size = 4) %>%
  layer_flatten() %>%
  layer_dense(32, activation = "relu") %>%
  layer_dense(6, activation = "softmax")

model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = "adam",
  metrics = "accuracy"
)

history <- model %>%
  fit(x_train, train$y, batch_size = 50, epochs = 30, verbose = 0, validation_split = 0.3)
plot(history)
results <- model %>% evaluate(x_test, test$y, batch_size=50, verbose = 2)
results

#### Pre-trained embeddings ####
set.seed(321)
sample_index <- createDataPartition(sona$president_13, p = 0.7, list = FALSE)

training_rows <- sample_index

train <- list()
test <- list()
train$x <- sona$speech[training_rows]
test$x <-  sona$speech[-training_rows]

train$y <- to_categorical(as.integer(sona$president_13[training_rows]) - 1)
test$y <-  to_categorical(as.integer(sona$president_13[-training_rows]) - 1)


embedding <- "https://tfhub.dev/google/nnlm-en-dim50/2"
hub_layer <- tfhub::layer_hub(handle = embedding, trainable = TRUE)
hub_layer(train$x)

tensorflow::set_random_seed(4)
model <- keras_model_sequential() %>%
  hub_layer() %>% 
  layer_dense(units = 16, activation = "relu") %>% 
  layer_dropout(rate=0.2) %>% 
  layer_dense(units = 6, activation = "softmax")

model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = "adam",
  metrics = "accuracy"
)

history <- model %>% fit(train$x, train$y, epochs = 10, 
                         batch_size = 64,  validation_split = 0.1, verbose = 0) 
plot(history)

results <- model %>% evaluate(test$x, test$y, batch_size = 20, verbose = 2)
results
