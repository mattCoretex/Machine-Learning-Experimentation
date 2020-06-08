# CA-real-estate-prediction

Predict real estate price range from real estate listing using two Neural Network models. <br>
The first model is a Scikit-Learn MLP classifier and the other is Keras Sequential.

## Dataset:
+ contains 38,784 Canadian Real Estate listing
+ has 9 features:
	+ bath, bed, city, postal code, price, property type, sqft, stories, and year built.
+ is preprocessed within the Jupyter Notebook

## Encoded Price Range (predicting values):
+ label 1 denotes property price under $200,000
+ label 2 denotes property price between $200,000 - $399,999
+ label 3 denotes property price between $400,000 - $799,999
+ label 4 denotes property price over $800,000

## Note: 
+ The price range is chosen based on the best data distribution when grouping property prices (see in Jupyter Notebook)
