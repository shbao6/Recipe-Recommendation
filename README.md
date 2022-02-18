# Recipe-Recommendation
There is a similarity-based recipe recommendation. 
In this project, I build recommender systems to make predictions related to user/recipe interactions from Food.com.

My first step was to solve for imbalanced data issue. The validation set only consists of positive samples but I also need examples of user/item pairs corresponding to recipes that weren't cooked. I sample a negative entry by randomly choosing a recipe that user hasn't cooked. The validation set then has 200,000 pairs of user and recipe, with cooked and non-cooked split evenly.

About Data Features and models

1.1 Item/User Popularity 

There are many ways to construct the indicator of popularity. For instance, someone can sort all recipes (users) by their times of occurrence and use the ranking as the indicator. People can also draw a line between popular recipes and unpopular ones, and use a Boolean value as the indicator. Directly using the value of occurrence times is another way to deal with the data. When it comes to the competition, I find the value-based system is good to use.

1.2 Similarity

The similarity is used to describe the connection between items and users. I have tried two ways to measure the similarity, which are the Jaccard similarity and the Cosine similarity. In the validation set, Jaccard similarity does slightly better than its counterpart.

1.3 Optimized Model

To improve the predictor, I incorporated the Jaccard-based threshold and the popularity-based threshold. I converted the popularity into percentile numbers and stacked it with the similarity matrix to obtain a large feature matrix. Then applied logistic regression to train the model and used the model to predict test data.

2.1 Rating Prediction

I used the Simple bias only latent factor-based recommender. I found this approach works better than Factorization Machine (fastFM) and works equally good as the latent factor model (surprise). The package I used is scipy.optimize.

2.2 Utility Structure

I build some utility data structures to store the variables of our model (alpha, userBiases, and itemBiases). The actual prediction function of my model is simple: Just predict using a global offset (alpha), a user offset (beta_u in the slides), and an item offset (beta_i).

2.3 Optimize Cost Function

To perform gradient descent, the library I used requires passing it a "flat" parameter vector (theta) containing all parameters. I defined a utility function which converts between a flat feature vector and my model parameters, i.e., it "unpacks" theta into the offset and bias parameters.
The "cost" function is the function I’m trying to optimize. This is a requirement of the gradient descent library. In this case, I computed the (regularized) MSE of a particular solution (theta) and returning the cost.
Then I can run the gradient descent. The inputs were: the cost function; initial parameter values; derivative function defined; the ratings in the training set and the regularization strength lambda. The model will determine the best values of the parameters when stopping at the smallest MSE value.

2.4 Prediction Function

I defined the prediction function as alpha + userBiases[user] + itemBiases[item]. Alpha was the average rating in the training data. Then I applied the prediction function on the validation set and got MSEs with different lambdas. The best lambda with the smallest MSE was 0.00001.
With the best lambda = 0.00001 in mind, I trained the model on the entire dataset with 500,000 reviews to get the new parameters. Alpha became the average rating in the entire dataset.
Finally, I obtained the predictions for testing data using the prediction function with new parameters. For each pair of (user, recipe), if the recipe is new, it will return Alpha + userBiases; if the user is new, it will return Alpha + itemBiases; if they are all new, it will return Alpha only. For the rest (user, recipe) pairs, it will return Alpha + userbiases + itemBiases.
