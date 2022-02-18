# Recipe-Recommendation
There is a similarity-based recipe recommendation. 
In this project, I build recommender systems to make predictions related to user/recipe interactions from Food.com.

My first step is to solve for imbalanced data issue. The validation set only consists of positive samples but I also need examples of user/item pairs corresponding to recipes that weren't cooked. I sample a negative entry by randomly choosing a recipe that user hasn't cooked. The validation set then has 200,000 pairs of user and recipe, with cooked and non-cooked split evenly.

About Data Features and models

1.1 Item/User Popularity 

There are many ways to construct the indicator of popularity. For instance, someone can sort all recipes (users) by their times of occurrence and use the ranking as the indicator. People can also draw a line between popular recipes and unpopular ones, and use a Boolean value as the indicator. Directly using the value of occurrence times is another way to deal with the data. When it comes to the competition, I find the value-based system is good to use.

1.2 Similarity

The similarity is used to describe the connection between items and users. I have tried two ways to measure the similarity, which are the Jaccard similarity and the Cosine similarity. In the validation set, Jaccard similarity does slightly better than its counterpart.

1.3 Optimized Model

To improve the predictor, I incorporate the Jaccard-based threshold and the popularity-based threshold. I convert the popularity into percentile numbers and stack it with the similarity matrix to obtain a large feature matrix. Then apply logistic regression to train the model and use the model to predict test data.

2.1 Rating Prediction

I use the Simple Bias Latent Factor-based recommender. I find this approach works better than Factorization Machine (fastFM) and works equally good as the latent factor model (surprise). The package I used is scipy.optimize.

2.2 Utility Structure

I build some utility data structures to store the variables of our model (alpha, userBiases, and itemBiases). The actual prediction function of my model is simple: Just predict using a global offset (alpha), a user offset (beta_u in the slides), and an item offset (beta_i).

2.3 Optimize Cost Function

The "cost" function is the function I’m trying to optimize. I compute the (regularized) MSE of a particular solution (theta) and returning the cost.
The inputs are: the cost function; initial parameter values; derivative function defined; the ratings in the training set and the regularization strength lambda. The model will determine the best values of the parameters when stopping at the smallest MSE value.

2.4 Prediction Function

The prediction function is defined as alpha + userBiases[user] + itemBiases[item]. Alpha is the average rating in the training data. I apply the prediction function on the validation set and get the best lambda to train the model on the entire dataset with 500,000 reviews. For each pair of (user, recipe), if the recipe is new, it will return Alpha + userBiases; if the user is new, it will return Alpha + itemBiases; if they are all new, it will return Alpha only. For the rest (user, recipe) pairs, it will return Alpha + userbiases + itemBiases.
