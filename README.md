# spark-movies-als

This is a sample application demonstrates the ALS algorithm - Spark's MLlib implements a collaborative filtering algorithm. ALS models the rating matrix (R) as the multiplication of a low-rank user (U) and product (V) factors, and learns these factors by minimizing the reconstruction error of the observed ratings. The unknown ratings can subsequently be computed by multiplying these factors. In this way, we can recommend products based on the predicted ratings.

The code is self-explained, but I added a few remarks. I printed the recommended movies for a particular user. Then the next printing is the actual vs the predicted ratings. At you can see the MSE - Mean Squared Error. MSE is the difference between the predicted and the actual target

In order to run it, you need to download the movies data dile from here : http://files.grouplens.org/datasets/movielens/

Open the zip file and run the program by supplying the movies file as first parameter and rating file as the second parameter.

