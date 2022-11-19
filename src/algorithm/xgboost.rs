use crate::objects::DataFrame;

pub struct XGBoost {
    learning_rate: f64,
    n_estimators: i32,
    regularisation: f64,
}

impl XGBoost {
    pub fn new(learning_rate: f64, n_estimators: i32, regularisation: f64) -> XGBoost {
        XGBoost {
            learning_rate,
            n_estimators,
            regularisation,
        }
    }

    pub fn train(
        &self,
        training_data: &DataFrame,
        target_column: &str,
        feature_columns: Vec<&str>,
    ) -> Result<bool, String> {
        let y_train = training_data.get_column(target_column);
        let X_train = training_data.get_columns(feature_columns);

        // Initial prediction is average of y_train
        let initial_prediction = DataFrame::average(&y_train);

        // Calculate residuals
        let residuals_result = &self.residuals(&y_train, &vec![initial_prediction; y_train.len()]);

        let residuals = match residuals_result {
            Ok(val) => val,
            Err(error) => return Err(error.to_string())
        };

        // Calculate similarity score
        let similarity = &self.similarity_score(residuals);

        Ok(true)
    }

    fn similarity_score(&self, residuals: &Vec<f64>) -> f64{
        let sum = residuals.iter().sum::<f64>();

        let score = (sum*sum)/(residuals.len() as f64 + &self.regularisation);

        score
    }

    fn residuals(&self,
        actual_values: &Vec<f64>,
        predicted_values: &Vec<f64>,
    ) -> Result<Vec<f64>, String> {
        if actual_values.len() != predicted_values.len() {
            return Err(format!("Mismatched vector length. actual_values length is {}. predicted_values length is {}",actual_values.len(),predicted_values.len()));
        }

        let mut residuals = vec![];
        for i in 0..actual_values.len() {
            let residual = actual_values[i] - predicted_values[i];
            residuals.push(residual);
        }

        Ok(residuals)
    }
}
