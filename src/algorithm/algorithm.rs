use std::mem;

use super::node::Node;
use crate::objects::DataFrame;

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
        feature_columns: &Vec<String>,
    ) -> Result<bool, String> {
        let y_train = training_data.get_column(target_column);
        let X_train = training_data.get_columns(feature_columns);

        // Initial prediction is average of y_train
        let initial_prediction = DataFrame::average(&y_train);

        // Calculate initial residuals
        let residuals_result = &self.residuals(&y_train, &vec![initial_prediction; y_train.len()]);

        let residuals = match residuals_result {
            Ok(val) => val,
            Err(error) => return Err(error.to_string()),
        };

        // For each column, create a root node
        for i in 0..feature_columns.len() {
            // TODO: Need to calculate best split for each column, then the best column to split the root by
            //let values = X_train.get_column(feature_columns[i]);
            //let root_node = calculate_best_split(, residuals, regularisation, column_index)
        }

        Ok(true)
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

fn calculate_best_split(
    values: &Vec<f32>,
    residuals: &Vec<f32>,
    regularisation: f32,
    column_index: i32,
) -> Node {
    // Sort values
    let (sorted_values, sorted_residuals) = sort_multiple(values, residuals);

    // Find average of each of the adjacent values
    let mut adjacent_averages = vec![];
    for i in 0..sorted_values.len() {
        if i == sorted_values.len() - 1 {
            break;
        } else {
            let adjacent_average = (sorted_values[i] + sorted_values[i + 1]) / 2.0;
            adjacent_averages.push(adjacent_average);
        }
    }

    // For each adjacent average, create a node split
    let mut root_nodes = vec![];
    for adjacent_average in adjacent_averages {
        // Get indices of the values that are left and right of the split
        let mut left_leaf_indices = vec![];
        let mut right_leaf_indices = vec![];
        for i in 0..sorted_values.len() {
            if sorted_values[i] <= adjacent_average {
                left_leaf_indices.push(i);
            } else {
                right_leaf_indices.push(i);
            }
        }

        // From those indices, get the residuals
        let mut left_leaf_residuals = vec![];
        let mut right_leaf_residuals = vec![];
        for index in left_leaf_indices.iter() {
            left_leaf_residuals.push(residuals[*index]);
        }
        for index in right_leaf_indices.iter() {
            right_leaf_residuals.push(residuals[*index]);
        }

        let left_node = Node::new(None, None, left_leaf_residuals, regularisation);
        let right_node = Node::new(None, None, right_leaf_residuals, regularisation);
        let root_node = Node::new(
            Some(column_index),
            Some(adjacent_average),
            sorted_residuals.clone(),
            regularisation,
        );

        root_nodes.push(root_node);
    }

    // Select the branch with the best gain
    let mut best_index = 0;
    let mut best_gain = root_nodes[0].gain();
    for i in 0..root_nodes.len() {
        if i == 0 {
            continue;
        }

        let gain = root_nodes[i].gain();

        if gain > best_gain {
            best_index = i;
            best_gain = gain;            
        }
    }

    let best_node = mem::replace(&mut root_nodes[best_index], Node::new(None,None,vec![],0.0));
    best_node
}

/// Given two vectors, sorts vec1, and then reorganises vec2 by the new order of vec1
fn sort_multiple(vec1: &Vec<f32>, vec2: &Vec<f32>) -> (Vec<f32>, Vec<f32>) {
    // TODO: implementation
    (vec1.clone(), vec2.clone())
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
