######    info   ########
#
# Purposee: Example NLP Script
# Data    : https://www.kaggle.com/datasets/tarique7/airline-incidents-safety-data
#

######    packages   ########
# install.packages("reticulate")# python
# install.packages("tidyverse")  # data cleaning
# install.packages("ggplot2")    # vis
# install.packages("tm")         # nlp / corpus
# install.packages("regex")      # regular expression
# install.packages("syuzhet")     # sentiment
# install.packages("topicmodels")# topic modeling
# install.packages("udpipe")     # udpipe
# install.packages("igraph")     # igraph
# install.packages("DataExplorer")     # DataExplorer EDA
# install.packages("janitor")     # fix col names for R

##  packag info links
#   https://cran.r-project.org/web/packages/DataExplorer/vignettes/dataexplorer-intro.html
#   https://stringr.tidyverse.org/reference/str_trim.html
#   https://cran.r-project.org/web/packages/textstem/readme/README.html


## free book info
#   https://www.tidytextmining.com/index.html

######    library   ########
library(tidyverse)
library(ggplot2)
library(tm)
library(regex)
library(syuzhet)
library(topicmodels)
library(udpipe)
library(igraph)
library(DataExplorer)
library(stringr) # working with strings
library(udpipe)
library(data.table)
udmodel <- udpipe_download_model(language = "english")
udmodel_en <- udpipe_load_model(file = udmodel$file_model)
######    get data   ########

# read in
data                                                <- fread("Airline Occurences.csv")

# fix columns
setnames(x=data, old=names(data), new=gsub(" ","_",names(data)))
names(data) <- names(data) %>% tolower()


# remove trailing and leading spaces 
data$report                                         <- str_trim(data$report , side = "both")
data$part_failure                                   <- str_trim(data$part_failure , side = "both")
data$occurence_nature_condition                     <- str_trim(data$occurence_nature_condition , side = "both")
data$occurence_precautionary_procedures             <- str_trim(data$occurence_precautionary_procedures , side = "both")

# add index to join back
data <- data %>% mutate(id = row_number(report))

######    eda   ########

#Ranking of Maths Score by Countries

plot_str(data) #liniage
plot_intro(data) # quick missing data check
summary(data) #distributions


# Part.Failure  
# EMERGENCY LIGHTS INOPERATIVE     :  315  
# ZONE 200 - FUSE INOPERATIVE      :  289  
# UNKNOWN UNKNOWN                  :  277  
# LT ELEVATOR EXCESS PLAY          :  221  
# ZONE 100 - FUSE CORRODED         :  218  
# RT ELEVATOR EXCESS PLAY          :  205  
# (Other)                          :14373 
#plot_bar(data) #bar plot on categories

# look at the data w/out other
data_prec_o    <- data %>% filter(occurence_precautionary_procedures != 'OTHER')
data_prec_none <- data_prec_o %>% filter(occurence_precautionary_procedures != "NONE")

#MECE - no NONE OR OTHER, and ONLY NONE AND OTHER
data_prec_not_o_n <- data %>% dplyr::filter(occurence_precautionary_procedures != 'NONE' & occurence_precautionary_procedures != 'OTHER')
data_prec_only <- data %>% dplyr::filter(occurence_precautionary_procedures == 'NONE' | occurence_precautionary_procedures == 'OTHER')

# barchart
plot_bar(data_prec_not_o_n) #bar plot on categories
plot_bar(data_prec_only) #bar plot on categories

#remove data from env which we won't use
rm(data_prec_none)
rm(data_prec_o)

######    hypothesis / questions   ########

# 1. Why do we have so many categories which are 'none', and 'other?
# 2. Should we focus categories which do not have an resolution?
# 3. There are many categories for parts - do we have a part taxonomy to get to a bigge issue?
# 4. Topic / cluster to find new issues?

# How to decide?
#   -- Talk to the business
#   -- Reccomend to split out due to unbalance 

######    toy example first: create a corpus + tdm   ########

# start small: 
set.seed(12345)                             # Set seed for reproducibility
ct <- nrow(data_prec_only) *.3
data_prec_only <- sample_n(data_prec_only, ct)                # Sample rows of data with dplyr

######    get lemma and remove punct   ########

library(tm)
# 
# #quick example of annotation ahead of cleaning the text 
# x <- udpipe_annotate(udmodel_en, x = data_prec_only$report[1])
# x <- as.data.frame(x)
# str(x)
# table(x$upos)

#Example before
b <- data_prec_only$report[12]

library(textstem)
cor_other <- lemmatize_strings(data_prec_only$report)
cor       <- lemmatize_strings(data_prec_not_o_n$report)

c <- cor_other[12]

# print out before and after
b
c
s
# Observatioons
# what changed? 
# fitting became fit
# acomplished became acomplish

# was this a good change?
# - depends - when dealing wich mechanical parts, they often lose meaning 
# during stemming / lemma because a fitting means something different


######    stopwords, cleaning, dtm   ########
# custom stopword
stopwords_cust <- c("hello","world")



#Create a vector containing only the text & Create a corpus
cor_other <-  Corpus(VectorSource(cor_other))
cor       <-  Corpus(VectorSource(cor))



# clean data function 
clean_data <- function(docs) {
  
  docs <- docs %>%
    tm_map(removeNumbers) %>%
    tm_map(removePunctuation) %>%
    tm_map(stripWhitespace)
  docs <- tm_map(docs, content_transformer(tolower))
  docs <- tm_map(docs, removeWords, stopwords("english"))
  docs <- tm_map(docs, removeWords, stopwords_cust)
}

# apply cleaning the data
cor_other_c  <-  clean_data(cor_other)
cor_c        <-  clean_data(cor)


# Create a document-term-matrix function
dtm_data_df <- function(docs) {
  dtm <- TermDocumentMatrix(docs) 
  matrix <- as.matrix(dtm) 
  words <- sort(rowSums(matrix),decreasing=TRUE) 
  df <- data.frame(word = names(words),freq=words)
}


# create doc - term - matrix
memory.limit(size=50)
cor_other_df  <-  dtm_data_df(cor_other_c)
cor_df        <-  dtm_data_df(cor_c)

head(cor_dtm)
head(cor_dtm)
# word freq
# amm           amm 5109
# check       check 3678
# replace   replace 3641
# engine     engine 3610
# aircraft aircraft 3538
# per           per 3270

## You can leverage these as 'topics' or to understand 'stopwords' appear first

######    word cloud   ########
library(wordcloud)
set.seed(1234) # for reproducibility 
wordcloud(words = cor_other_df$word, freq = cor_other_df$freq, min.freq = 100
          , max.words=100
          , random.order=FALSE, rot.per=0.35
          , colors=brewer.pal(8, "Dark2"))

wordcloud(words = cor_df$word, freq = cor_df$freq, min.freq = 100
          , max.words=100
          , random.order=FALSE, rot.per=0.35
          , colors=brewer.pal(8, "Dark2"))

# word clouds are 'different', but not as useful
# Let's try looking at top 1, 2, and topics
# Good resource to follow up wih https://uc-r.github.io/word_relationships#visualize
# https://uc-r.github.io/creating-text-features (dlya Ani posmotret potom)



# plot sentiment
library(syuzhet)



######    topic modeling   ########
library(topicmodels) # works from dtm

#dtm
cor_other_dtm  <-  DocumentTermMatrix(cor_other)
cor_dtm        <-  DocumentTermMatrix(cor)
# set a seed so that the output of the model is predictable
cor_other_lda <- LDA(cor_other_dtm, k = 15, control = list(seed = 1234))
cor_lda <- LDA(cor_other_dtm, k = 15, control = list(seed = 1234))

# CTM 
cor_other_ctm <- CTM(cor_other_dtm, k = 15, control = list(seed = 1234))
cor_ctm  <- CTM(cor_other_dtm, k = 15, control = list(seed = 1234))

print(cor_other_ctm)
# Save an object
saveRDS(cor_other_lda, "cor_other_lda.rds")
saveRDS(cor_lda, "cor_lda.rds")
saveRDS(cor_other_ctm, "cor_other_ctm.rds")
saveRDS(cor_ctm, "cor_ctm.rds")
save.image(file="3.19.23.RData") 


terms(cor_other_lda) # Observation that 'and' should be a stopword :)
terms(cor_lda)       # Observation that 'and','the', 'per' should be a stopword :)
terms(cor_other_ctm) # Observe different top words comming up
terms(cor_ctm)
#print(lda_model.print_topics())

## Python: https://www.machinelearningplus.com/nlp/topic-modeling-visualization-how-to-present-results-lda-models/



######    top 3 words by document   ########
# https://stackoverflow.com/questions/23655694/r-find-most-frequent-group-of-words-in-corpus
# https://sicss.io/2020/materials/day3-text-analysis/basic-text-analysis/rmarkdown/Basic_Text_Analysis_in_R.html




