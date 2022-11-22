use crate::objects::DataFrame;

/// A node with no children is a leaf
struct Node {
    column_index: i32,
    residuals: Vec<f32>,
    regularisation: f32,
    left_child: Option<Box<Node>>,
    right_child: Option<Box<Node>>,
}

impl Node {
    pub fn new(
        column_index: i32,
        residuals: Vec<f32>,
        regularisation: f32,
        left_child: Option<Box<Node>>,
        right_child: Option<Box<Node>>,
    ) -> Self {
        Node {
            column_index,
            residuals,
            regularisation,
            left_child,
            right_child,
        }
    }

    /// The similarity score is the sum of the residuals squared, divided by the number of residuals plus the regularisation parameter
    pub fn similarity_score(&self) -> f32 {
        let sum = self.residuals.iter().sum::<f32>();

        let score = (sum * sum) / (self.residuals.len() as f32 + self.regularisation);

        score
    }

    // Calculates the gain of the child split
    pub fn gain(&self) -> f64 {
        let gain = match &self.parent_branch {
            Some(parent_branch) => {
                let parent_leaf: &Leaf;

                if self.parent_is_left {
                    parent_leaf = &parent_branch.left_leaf;
                } else {
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

pub struct XGBoost {
    learning_rate: f64,
    n_estimators: i32,
    regularisation: f64,
    max_tree_depth: i32,
}

impl XGBoost {
    pub fn new(
        learning_rate: f64,
        n_estimators: i32,
        regularisation: f64,
        max_tree_depth: i32,
    ) -> Self {
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

        // // Calculate residuals
        // let residuals_result = &self.residuals(&y_train, &vec![initial_prediction; y_train.len()]);

        // let residuals = match residuals_result {
        //     Ok(val) => val,
        //     Err(error) => return Err(error.to_string()),
        // };

        // // Calculate similarity score
        // let similarity = &self.similarity_score(residuals);

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

// pub fn calculate_best_split(
//     &self,
//     values: &Vec<f64>,
//     residuals: &Vec<f64>,
//     column_name: &str,
// ) -> Branch {
//     // TODO: sort the values - is a pain due to rust not being able to sort floats - may need quicksort implementation

//     // First, find average of each of the adjacent values
//     let mut adjacent_averages = vec![];
//     for i in 0..values.len() {
//         if i == values.len() - 1 {
//             break;
//         } else {
//             let adjacent_average = (values[i] + values[i + 1]) / 2.0;
//             adjacent_averages.push(adjacent_average);
//         }
//     }

//     // For each adjacent average, create a branch
//     let mut branches = vec![];
//     for adjacent_average in adjacent_averages {
//         let mut left_leaf_indices = vec![];
//         let mut right_leaf_indices = vec![];
//         for i in 0..values.len() {
//             if values[i] <= adjacent_average {
//                 left_leaf_indices.push(i);
//             } else {
//                 right_leaf_indices.push(i);
//             }
//         }

//         let mut left_leaf_residuals = vec![];
//         let mut right_leaf_residuals = vec![];
//         for index in left_leaf_indices.iter() {
//             left_leaf_residuals.push(residuals[*index]);
//         }
//         for index in right_leaf_indices.iter() {
//             right_leaf_residuals.push(residuals[*index]);
//         }

//         let left_leaf = Leaf::new(left_leaf_residuals, self.regularisation, column_name.to_string());

//         let right_leaf = Leaf::new(right_leaf_residuals, self.regularisation, column_name.to_string());

//         let branch = Branch::new(self.parent_branch, self.parent_is_left, left_leaf, Some(right_leaf), self.regularisation);

//         branches.push(branch);
//     }

//     // Select the branch with the best gain
//     let mut best_index = 0;
//     let mut best_gain = branches[0].gain();
//     for i in 0..branches.len() {
//         if i == 0{
//             continue;
//         }

//         let gain = branches[i].gain();

//         if gain > best_gain{
//             best_index = i;
//         }
//     }

//     return branches[best_index];
// }

// /// The gain is the sum of the similarity scores of the leaves in the branch, minus the similarity score of the parent leaf.
// pub fn gain(&self) -> f64 {
//     let gain = match &self.parent_branch {
//         Some(parent_branch) => {
//             let parent_leaf: &Leaf;

//             if self.parent_is_left {
//                 parent_leaf = &parent_branch.left_leaf;
//             } else {
//                 parent_leaf = match &parent_branch.right_leaf {
//                     Some(leaf) => leaf,
//                     None => return 0.0,
//                 };
//             }

//             let right_leaf = match &self.right_leaf {
//                 Some(leaf) => leaf,
//                 None => return 0.0,
//             };

//             // left leaf similarity + right leaf similarity - parent similarity
//             self.left_leaf.similarity_score() + right_leaf.similarity_score()
//                 - parent_leaf.similarity_score()
//         }

//         None => 0.0,
//     };

//     gain
// }
