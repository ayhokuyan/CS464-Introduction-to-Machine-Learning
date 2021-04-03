import pandas as pd
import numpy as np

#get the argmax of each prediction list from list of lists and count matches, then print the accuracy
def calcPerc(test_pred, y_test_dt):
    true_count = 0
    false_count = 0
    count = -1
    for x in test_pred:
        max_index = np.argmax(x)
        count += 1
        if max_index == 0 and y_test_dt.iloc[count].values == 'neutral':
            true_count += 1
        elif max_index == 1 and y_test_dt.iloc[count].values == 'positive':
            true_count += 1
        elif max_index == 2 and y_test_dt.iloc[count].values == 'negative':
            true_count += 1
        else:
            false_count += 1
    print('Number of correct predictions: ' + str(true_count))
    print('Number of false predictions: ' + str(false_count))
    print('Test accuracy: ' + str(100*true_count/(true_count+false_count)) + '%')

# read training csv files and the txt file
nms_frq = pd.read_csv('question-4-vocab.txt', header=None, sep='\t', names=['names', 'freqs'])
x_train = pd.read_csv('question-4-train-features.csv', header=None, sep=',', names=nms_frq['names'])
y_train = pd.read_csv('question-4-train-labels.csv', header=None, sep=',', names=['results'])

# concatenate features and labels for easy grouping
train = pd.concat([x_train, y_train], axis=1)

# set prior probabilities
# get the negative priors for each word
neg_prior = train.loc[train['results'] == 'negative'].select_dtypes(pd.np.number).sum().rename('total_neg')
# get the positive priors
pos_prior = train.loc[train['results'] == 'positive'].select_dtypes(pd.np.number).sum().rename('total_pos')
# get the neutral priors
ntr_prior = train.loc[train['results'] == 'neutral'].select_dtypes(pd.np.number).sum().rename('total_ntr')

total_prior = x_train.select_dtypes(pd.np.number).sum().rename('total_prior')

# get the total number of words in each res=neg/pos/ntr matrix
total_pos_word_cnt = train.loc[train['results'] == 'positive'].select_dtypes(pd.np.number).values.sum()
total_neg_word_cnt = train.loc[train['results'] == 'negative'].select_dtypes(pd.np.number).values.sum()
total_ntr_word_cnt = train.loc[train['results'] == 'neutral'].select_dtypes(pd.np.number).values.sum()

# get prior probabilities of labels
pr_prob_pos = train.loc[train['results'] == 'positive'].shape[0] / x_train.shape[0]
pr_prob_neg = train.loc[train['results'] == 'negative'].shape[0] / x_train.shape[0]
pr_prob_ntr = 1 - (pr_prob_pos + pr_prob_neg)

# get the ln for each label prior (used in a lot of calculations)
pr_prob_pos_ln = np.log(pr_prob_pos)
pr_prob_neg_ln = np.log(pr_prob_neg)
pr_prob_ntr_ln = np.log(pr_prob_ntr)

neg_matrix = train.loc[train['results'] == 'negative']
print('Number of negative tweets in the training data: ' + str(neg_matrix.shape[0]))
print('Total number of tweets: ' + str(train.shape[0]))
print('Percentage of negative tweets: ' + str(neg_matrix.shape[0]/train.shape[0]))

# read csv files for test data
x_test = pd.read_csv('question-4-test-features.csv', header=None, sep=',', names=nms_frq['names'])
y_test = pd.read_csv('question-4-test-labels.csv', header=None, sep=',', names=['results'])


# ## Multinomial Naive Bayes Model

# ML Estimation

# calculate the probabilities for each word in any label
neg_prior_prb = np.log(neg_prior/total_neg_word_cnt)
pos_prior_prb = np.log(pos_prior/total_pos_word_cnt)
ntr_prior_prb = np.log(ntr_prior/total_ntr_word_cnt)

# run the prediction for each test sample
mle_test_prediction = [[None for y in range(3)] for x in range(y_test.shape[0])]
count_y_ntr = 0
count_y_pos = 1
count_y_neg = 2
count_x = 0
for i in range(x_test.shape[0]):
    row = x_test.iloc[i]
    temp_pos = pos_prior_prb * row.values  
    temp_pos = pr_prob_pos_ln + np.sum(temp_pos.fillna(0))
    temp_neg = neg_prior_prb * row.values
    temp_neg = pr_prob_neg_ln + np.sum(temp_neg.fillna(0))
    temp_ntr = ntr_prior_prb * row.values
    temp_ntr = pr_prob_ntr_ln + np.sum(temp_ntr.fillna(0))
    mle_test_prediction[count_x][count_y_ntr] = temp_ntr
    mle_test_prediction[count_x][count_y_pos] = temp_pos
    mle_test_prediction[count_x][count_y_neg] = temp_neg
    count_x += 1              

print("Multinomial Naive Bayes Classifier with ML Estimator\n")
calcPerc(mle_test_prediction, y_test)
print("\n")


# MAP Estimation

# calculate the probabilities for each word in any label given alpha = 1
alpha = 1
neg_prior_prb = np.log((neg_prior+alpha)/(total_neg_word_cnt+(alpha*x_train.shape[1])))
pos_prior_prb = np.log((pos_prior+alpha)/(total_pos_word_cnt+(alpha*x_train.shape[1])))
ntr_prior_prb = np.log((ntr_prior+alpha)/(total_ntr_word_cnt+(alpha*x_train.shape[1])))

# run the prediction for each test sample
map_test_prediction = [[None for y in range(3)] for x in range(y_test.shape[0])]
count_y_ntr = 0
count_y_pos = 1
count_y_neg = 2
count_x = 0

for i in range(x_test.shape[0]):
    row = x_test.iloc[i]
    temp_pos = pos_prior_prb * row.values  
    temp_pos = pr_prob_pos_ln + np.sum(temp_pos.fillna(0))
    temp_neg = neg_prior_prb * row.values
    temp_neg = pr_prob_neg_ln + np.sum(temp_neg.fillna(0))
    temp_ntr = ntr_prior_prb * row.values
    temp_ntr = pr_prob_ntr_ln + np.sum(temp_ntr.fillna(0))
    map_test_prediction[count_x][count_y_ntr] = temp_ntr
    map_test_prediction[count_x][count_y_pos] = temp_pos
    map_test_prediction[count_x][count_y_neg] = temp_neg
    count_x += 1              

print("Multinomial Naive Bayes Classifier with MAP Estimator\n")
calcPerc(map_test_prediction,y_test)
print("\n")

# ## Bernoulli Naive Bayes Model

# convert training data to 1/0s for the bernoulli classifier
x_train_br = x_train.copy()
x_train_br[x_train_br != 0] = 1

# concatenate features and labels for easy grouping
train_br = pd.concat([x_train_br, y_train], axis=1)

# set prior probabilities
# get the negative priors for each word
neg_prior_br = train_br.loc[train['results'] == 'negative'].select_dtypes(pd.np.number).sum().rename('total_neg')
# get the positive priors
pos_prior_br = train_br.loc[train['results'] == 'positive'].select_dtypes(pd.np.number).sum().rename('total_pos')
# get the neutral priors
ntr_prior_br = train_br.loc[train['results'] == 'neutral'].select_dtypes(pd.np.number).sum().rename('total_ntr')

total_prior_br = x_train_br.select_dtypes(pd.np.number).sum().rename('total_prior')

# total number of pos/neg/ntr tweets
total_pos_tweets = train_br.loc[train['results'] == 'positive'].shape[0]
total_neg_tweets = train_br.loc[train['results'] == 'negative'].shape[0]
total_ntr_tweets = train_br.loc[train['results'] == 'neutral'].shape[0]

# get prior probabilities of labels
pr_prob_pos_br = train_br.loc[train['results'] == 'positive'].shape[0] / x_train_br.shape[0]
pr_prob_neg_br = train_br.loc[train['results'] == 'negative'].shape[0] / x_train_br.shape[0]
pr_prob_ntr_br = 1 - (pr_prob_pos + pr_prob_neg)

# get the ln for each label prior (used in a lot of calculations)
pr_prob_pos_ln = np.log(pr_prob_pos_br)
pr_prob_neg_ln = np.log(pr_prob_neg_br)
pr_prob_ntr_ln = np.log(pr_prob_ntr_br)


# transform the test features to 1/0s
x_test_br = x_test.copy()
x_test_br[x_test_br != 0] = 1

#calcluate the probabilities for each word and also their compliments
neg_prior_prb = neg_prior_br/total_neg_tweets
pos_prior_prb = pos_prior_br/total_pos_tweets
ntr_prior_prb = ntr_prior_br/total_ntr_tweets
neg_prior_prb_cmp = 1-neg_prior_prb
pos_prior_prb_cmp = 1-pos_prior_prb
ntr_prior_prb_cmp = 1-ntr_prior_prb

#run the prediction for each test sample
mle_test_prediction_br = [[None for y in range(3)] for x in range(y_test.shape[0])]
count_y_ntr = 0
count_y_pos = 1
count_y_neg = 2
count_x = 0
for i in range(x_test_br.shape[0]):
    row = x_test_br.iloc[i]
    row_cmp = 1 - row
    temp_pos = pos_prior_prb * row.values + pos_prior_prb_cmp * row_cmp.values
    temp_pos = pr_prob_pos_ln + np.sum(np.log(temp_pos.fillna(0)))
    temp_neg = neg_prior_prb * row.values + neg_prior_prb_cmp * row_cmp.values
    temp_neg = pr_prob_neg_ln + np.sum(np.log(temp_neg.fillna(0)))
    temp_ntr = ntr_prior_prb * row.values + ntr_prior_prb_cmp * row_cmp.values
    temp_ntr = pr_prob_ntr_ln + np.sum(np.log(temp_ntr.fillna(0)))
    mle_test_prediction_br[count_x][count_y_ntr] = temp_ntr
    mle_test_prediction_br[count_x][count_y_pos] = temp_pos
    mle_test_prediction_br[count_x][count_y_neg] = temp_neg
    count_x += 1

print("Bernoulli Naive Bayes Classifier with MAP Estimator\n")
calcPerc(mle_test_prediction_br,y_test)
print("\n")


# ## Finding the Most Commonly Used Words

print('Most commonly used words in positive tweets:')
print(pos_prior.nlargest(n=20,keep='all'))
print('Most commonly used words in negative tweets:')
print(neg_prior.nlargest(n=20,keep='all'))
print('Most commonly used words in neutral tweets:')
print(ntr_prior.nlargest(n=20,keep='all'))

