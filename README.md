Audience Rating Prediction for Rotten Tomatoes Movie Dataset
Steps Implemented:

    Import Necessary Libraries:
        Importing required libraries such as pandas, numpy, sklearn, XGBoost, etc.

    Load Dataset:
        Load the Rotten Tomatoes movie dataset and convert it into a pandas DataFrame.

    Data Preprocessing:

        Handle Missing Values / Imputation:
            Replace missing numerical values with the mean of the respective feature.
            Replace missing textual values with "Unknown".
            Replace missing categorical values with "NaN".

        Encode Categorical Features:
            For features with fewer unique values (e.g., 'rating', 'tomatometer_status'), apply One-Hot Encoding.
            For features with more unique values (e.g., 'genre', 'directors', etc.), apply Frequency Encoding.

        Generate Word Embeddings for Textual Features:
            For the feature 'movie_info', tokenize each sentence, remove stopwords, and generate word embeddings using the Word2Vec model with an embedding size of 10.

    Feature Extraction:

        Sentiment Score:
            Create a new feature "sentiment_score" derived from 'critics_consensus' using the VADER sentiment analyzer.

        Date Features:
            Extract the following date-related features from 'in_theaters_date' and 'on_streaming_date':
                'in_theaters_year', 'in_theaters_month', 'in_theaters_day', 'in_theaters_weekday', 'in_theaters_is_weekend'
                'on_streaming_year', 'on_streaming_month', 'on_streaming_day', 'on_streaming_weekday', 'on_streaming_is_weekend'
                'days_between' (the difference between 'in_theaters_date' and 'on_streaming_date')

    Feature Selection Using Correlation Analysis:

        Calculate Pearson correlation of each numerical and categorical feature with the target column 'audience_rating'.

        Visualize the correlation matrix using a bar chart.

        Selected Features:
            Features with medium to high correlation with the target ('audience_rating') are selected. These include:
                'runtime_in_minutes', 'tomatometer_rating', 'tomatometer_count'
                'directors_Frequency', 'tomatometer_status_Certified Fresh', 'tomatometer_status_Fresh', 'tomatometer_status_Rotten'
                'sentiment_score', 'in_theaters_year', 'in_theaters_weekday', 'on_streaming_year', 'days_between'
                Additionally, the embeddings of the movie_info and critics_consensus columns are selected.

    Normalization:
        The following columns are normalized:
            'runtime_in_minutes', 'tomatometer_rating', 'tomatometer_count', 'directors_Frequency', 'in_theaters_weekday', 'days_between'.

    Model Training:

        Data Split:
            Split the dataset into training, validation, and test sets.

        Model Selection:
            Three models are compared and evaluated:
                Linear Regression
                Random Forest Regression
                XGBoost Regression
            The model with the least RMSE and the highest R-squared (R²) value is selected, which is XGBRegressor.

        Hyperparameter Tuning:
            Use Grid Search to tune hyperparameters and find the best set of hyperparameters for the XGBRegressor model.

    Model Pipeline:
        Design the model pipeline using the best model and optimal hyperparameters.
        Train the pipeline using the combined training and validation data.
        Evaluate the model on the test set and calculate the performance metrics.

Final Evaluation on Test Dataset:

    Mean Squared Error (MSE): 192.81
    Root Mean Squared Error (RMSE): 13.89
    R-squared (R²): 0.53
    Custom Testing Accuracy: 75.86%

Accuracy Calculation:

The accuracy can be calculated using the Mean Absolute Percentage Error (MAPE) formula:

The accuracy is then calculated as:

In this case, the accuracy is 75.86%.
