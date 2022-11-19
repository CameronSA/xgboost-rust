
struct XGBoost {
    learning_rate: f64,
    n_estimators: i32,
}

impl XGBoost {
    pub fn new(learning_rate: f64, n_estimators: i32) -> XGBoost {
        XGBoost {
            learning_rate,
            n_estimators,
        }
    }
}