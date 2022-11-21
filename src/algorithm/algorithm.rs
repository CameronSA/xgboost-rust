use crate::objects::DataFrame;

struct Leaf {
    residuals: Vec<f64>,
    regularisation: f64,
}

impl Leaf {
    pub fn new(residuals: Vec<f64>, regularisation: f64) -> Self {
        Leaf {
            residuals,
            regularisation,
        }
    }

    /// The similarity score is the sum of the residuals squared, divided by the number of residuals plus the regularisation parameter
    pub fn similarity_score(&self) -> f64 {
        let sum = self.residuals.iter().sum::<f64>();

        let score = (sum * sum) / (self.residuals.len() as f64 + &self.regularisation);

        score
    }
}

struct Branch {
    /// If this is the root, only the leaf leaf will be populated.
    /// If 'parent_is_left' is true, the parent leaf is the left leaf of the parent branch. Otherwise, the parent leaf is the right leaf of the parent branch
    parent_branch: Option<Box<Branch>>,
    parent_is_left: bool,
    left_leaf: Leaf,
    right_leaf: Option<Leaf>,
}

impl Branch {
    pub fn new(
        parent_branch: Option<Box<Branch>>,
        parent_is_left: bool,
        left_leaf: Leaf,
        right_leaf: Option<Leaf>,
    ) -> Self {
        Branch {
            parent_branch,
            parent_is_left,
            left_leaf,
            right_leaf,
        }
    }

    /// The gain is the sum of the similarity scores of the leaves in the branch, minus the similarity score of the parent leaf.
    pub fn gain(&self) -> f64 {
        let gain = match &self.parent_branch {
            Some(parent_branch) => {
                let parent_leaf: &Leaf;

                if self.parent_is_left {
                    parent_leaf = &parent_branch.left_leaf;
                }
                else{
                    parent_leaf = match &parent_branch.right_leaf {
                        Some(leaf) => leaf,
                        None => return 0.0,
                    };
                }
                
                let right_leaf = match &self.right_leaf {
                    Some(leaf) => leaf,
                    None => return 0.0,
                };
                
                // left leaf similarity + right leaf similarity - parent similarity
                self.left_leaf.similarity_score() + right_leaf.similarity_score()
                    - parent_leaf.similarity_score()
            }

            None => 0.0,
        };

        gain
    }
}

/// A decision tree consists of various levels or branches.
/// Each branch has a left leaf, an optional right leaf, and an optional parent branch.
/// If a branch is the first branch, i.e. the root, then it will have no parent branch and only the left leaf will be populated.
/// The depth of the tree is determined by the number of branches
struct DecisionTree 
{

}

pub struct XGBoost {
    learning_rate: f64,
    n_estimators: i32,
    regularisation: f64,
    max_tree_depth: i32,
}

impl XGBoost {
    pub fn new(learning_rate: f64, n_estimators: i32, regularisation: f64, max_tree_depth: i32) -> Self {
        XGBoost {
            learning_rate,
            n_estimators,
            regularisation,
            max_tree_depth,            
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
            Err(error) => return Err(error.to_string()),
        };

        // Calculate similarity score
        let similarity = &self.similarity_score(residuals);

        Ok(true)
    }

    fn similarity_score(&self, residuals: &Vec<f64>) -> f64 {
        let sum = residuals.iter().sum::<f64>();

        let score = (sum * sum) / (residuals.len() as f64 + &self.regularisation);

        score
    }

    fn residuals(
        &self,
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
