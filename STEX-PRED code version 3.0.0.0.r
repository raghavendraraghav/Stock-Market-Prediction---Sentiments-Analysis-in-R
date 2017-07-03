rm(list = ls())
install.packages("wordcloud")
install.packages("tm")  ## to get library of stop words to remove them
install.packages("text2vec") ## for vectorization of text using iton function
install.packages("SnowballC") ## to get stem dictionarry
install.packages("data.table") ## fast table read 
list.of.packages <- c("tm", "text2vec","SnowballC","data.table")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)


library(data.table)
Combined_News_DJIA <- fread("C:/Users/ragha/Downloads/Combined_News_DJIA.csv")
View(Combined_News_DJIA)
Combined_News_DJIA$document <- paste(Combined_News_DJIA$Top1,Combined_News_DJIA$Top2,Combined_News_DJIA$Top3,Combined_News_DJIA$Top4,
                                     Combined_News_DJIA$Top5,Combined_News_DJIA$Top6,Combined_News_DJIA$Top7,Combined_News_DJIA$Top8,
                                     Combined_News_DJIA$Top9,Combined_News_DJIA$Top10,Combined_News_DJIA$Top11,Combined_News_DJIA$Top12,
                                     Combined_News_DJIA$Top13,Combined_News_DJIA$Top14,Combined_News_DJIA$Top15,Combined_News_DJIA$Top16,
                                     Combined_News_DJIA$Top17,Combined_News_DJIA$Top18,Combined_News_DJIA$Top19,Combined_News_DJIA$Top20,
                                     Combined_News_DJIA$Top21,Combined_News_DJIA$Top22,Combined_News_DJIA$Top23,Combined_News_DJIA$Top24,
                                     Combined_News_DJIA$Top25,sep='. ')



dim(Combined_News_DJIA)



Combined_News_DJIA$Date_f <- as.Date(strptime(Combined_News_DJIA$Date,'%m/%d/%Y'))



month(Combined_News_DJIA$Date_f[1])



data<- Combined_News_DJIA[,c(29,2,28)]



preprocess = function(data)
{
data$document <- gsub('b"|b\'|\\\\|\\"', "", data$document)
 
data$document <- gsub("([<>])|[[:punct:]]", "\\1", data$document)



data$document <- gsub("[[:digit:]]", "", data$document)


# remove html links
data$document = gsub("http\\w+", "", data$document)

# remove unnecessary spaces


data$document = gsub("[ \t]{2,}", "", data$document)
data$document = gsub("^\\s+|\\s+$", "", data$document)

data$document = tolower(data$document)
return(data)
}


data = preprocess(data)


View(data)
head(data)
library(tm)

rm_words <- function(string, words) 
  {
  stopifnot(is.character(string), is.character(words))
  spltted <- strsplit(string, " ", fixed = TRUE) # fixed = TRUE for speedup
  vapply(spltted, function(x) paste(x[!tolower(x) %in% words], collapse = " "), character(1))
}


data$document = rm_words(data$document, tm::stopwords("en"))


library(SnowballC)
library(text2vec)

stem_tokenizer = function(x)
{
  
  token = word_tokenizer(x)  ## generate words documentwize
  return(lapply(token,SnowballC::wordStem,language="porter"))
  
}

View(data)

data$document_token =stem_tokenizer(data$document)

data<- setkey(data,Date_f)


View(data)





# Vectorization 
vectorise = function(token)
  
{
  it_train = itoken(token$document_token, 
                  ids = token$Date_f, 
                  progressbar = FALSE)



vocab = create_vocabulary(it_train,ngram = c(1L,5L))
vectorizer = vocab_vectorizer(vocab)
dtm_train = create_dtm(it_train, vectorizer)



# define tfidf model
tfidf = TfIdf$new()

# fit model to train data and transform train data with fitted model
dtm_train_tfidf = fit_transform(dtm_train, tfidf)
return(dtm_train_tfidf)
}


# Vocabulary creation
vocal = function(token)
  
{
  it_train = itoken(token$document_token, 
                    ids = token$Date_f, 
                    progressbar = FALSE)
  
  
  
  vocab = create_vocabulary(it_train,ngram = c(1L,5L))
 

  return(vocab)
}


data_train <- data[data$Date_f<="2014-12-31",]

data_test  <- data[data$Date_f>"2014-12-31",]


train_vec = vectorise(data_train)


test_vec = vectorise(data_test)

train_vocal = vocal(data_train)
test_vocal = vocal(data_test)



install.packages("glmnet")
library(glmnet)


#glmnet algo ----- Modeling 
glmnet_classifier = cv.glmnet(x = train_vec,
                              y = data_train$Label, 
                              family = 'binomial', 
                              alpha = 1,
                              type.measure = "auc",
                              nfolds = 5,
                              thresh = 1e-3,
                              maxit = 1e3)

fit_model = fit_transform(test_vec,model = glmnet_classifier, y = NULL)

predict(glmnet_classifier,type="coef")



summary(glmnet_classifier)

glmnet_classifier$glmnet.fit

plot(glmnet_classifier)

print(paste("max AUC =", round(max(glmnet_classifier$cvm), 4)))



