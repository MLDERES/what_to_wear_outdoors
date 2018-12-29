=======
History
=======

1.1.0 (2018-12-29)
------------------
- Implemented the strategy pattern for the ML predictors
    - This was important so that I could try to improve the machine learning models without having to deal with changing the different predictors for the sports
- Also, I made a few changes to the way the models work.
    - I generated about 100 random forecasts and set the proper outfits for them, this becomes the test data
    - I'm using the XLSX file, with actual outfits that I wore as the test platform
- Moved the scoring mechanisms for each of the models outside of the ML strategies as well so that I can play with how to determine how good the models really are.


1.0.0 (2018-12-17)
------------------
- Cleaned up the code-base quite a bit to make it more extensible
    - Created classes for predictors and translators
    - Leveraged a mix-in pattern to allow for a better sharing of common methods and definitions across classes
    - Took out dependencies between the classes,
        - For instance, I expect that the calling method will understand how to use weather forecast rather than
          having the predictor have to understand weather forecasts
        - Also require the caller to understand how to use the prediction to get a recommendation through the translator
          which should allow for different uses of the API rather than just doing the English response as
          initially provided

- Improved the learning model quite a bit, going from a logistic regression algorthim to a classifier
- Used two different models to determine those items that have multiple options (i.e. base layer) and those that only
  have two options (i.e. gloves or no gloves, heavy socks or regular socks)
- Setup functionality is working well
- Changed the raw data file format to more realistically deal with the items that have multiple choices
  (i.e. long sleeves, short-sleeves, None for base layers)



0.1.0 (2018-11-09)
------------------

* First release of the code
