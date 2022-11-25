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
        // Split the dataset into feature and target sets
        let y_train = match training_data.get_column(target_column) {
            Some(y_train) => y_train,
            None => return Err(format!("Target column {} not found", target_column)),
        };

        let x_train = match training_data.get_columns(feature_columns) {
            Some(x_train) => x_train,
            None => return Err(format!("Invalid feature columns: {:?}", feature_columns)),
        };

        // Check that all feature columns present
        let mut missing_feature_columns = vec![];
        for column_name in x_train.column_names() {
            let mut found = false;
            for feature_column in feature_columns {
                if feature_column == column_name {
                    found = true;
                    break;
                }
            }

            if !found {
                missing_feature_columns.push(column_name.to_string());
            }
        }

        if missing_feature_columns.len() != 0 {
            return Err(format!(
                "Training dataset is missing feature columns: {:?}",
                missing_feature_columns
            ));
        }

        // Initial prediction is average of y_train
        let initial_prediction = DataFrame::average(&y_train);

        // Calculate initial residuals
        let residuals_result = &self.residuals(&y_train, &vec![initial_prediction; y_train.len()]);

        let residuals = match residuals_result {
            Ok(val) => val,
            Err(error) => return Err(error.to_string()),
        };

        let root_node = self.build_decision_tree(&x_train, feature_columns, residuals);

        Ok(true)
    }

    fn build_decision_tree(
        &self,
        x_train: &DataFrame,
        feature_columns: &Vec<String>,
        residuals: &Vec<f64>,
    ) -> Result<Node, String> {
        let mut available_columns = feature_columns.clone();

        // Calculate root node
        let mut root_node = match self.compute_best_node(x_train, &available_columns, residuals) {
            Ok(val) => val,
            Err(error) => return Err(error.to_string()),
        };

        let root_column_index = match root_node.column_index() {
            Some(val) => val,
            None => return Ok(root_node),
        };

        // Whatever column was chosen for the root node, remove it from the column pool
        available_columns.remove(root_column_index);


        // NOTE: Just for the left (to get my head around it)
        // Will need multiple instances of 'available_columns' for each path through the tree

        // Iteration 1
        let root_node = match self.add_children(root_node, x_train, &available_columns) {
            Ok(val) => val,
            Err(error) => return Err(error.to_string()),
        };

        let left_node = match root_node.left_child(){
            Some(val) => val,
            None => return Ok(root_node),
        };

        let left_node_column_index = match left_node.column_index() {
            Some(val) => val,
            None => return Ok(root_node),
        };

        available_columns.remove(left_node_column_index);

        // Iteration 2
        let root_node = match self.add_children(*left_node.clone(), x_train, &available_columns) {
            Ok(val) => val,
            Err(error) => return Err(error.to_string()),
        };

        let left_left_node = match left_node.left_child(){
            Some(val) =>val,
            None => return Ok(root_node),
        };

        let left_left_node_column_index = match left_left_node.column_index() {
            Some(val) => val,
            None => return Ok(root_node),
        };

        available_columns.remove(left_left_node_column_index);

        // Iteration 3 
        // ...



        Ok(root_node)
    }

    fn add_children(
        &self,
        mut parent_node: Node,
        x_train: &DataFrame,
        available_columns: &Vec<String>,
    ) -> Result<Node, String> {
        // First the left
        let left_child_residuals = match parent_node.left_child() {
            Some(val) => val.residuals(),
            None => return Ok(parent_node),
        };

        let left_child =
            match self.compute_best_node(x_train, &available_columns, left_child_residuals) {
                Ok(val) => val,
                Err(error) => return Err(error.to_string()),
            };

        // Now the right
        let right_child_residuals = match parent_node.right_child() {
            Some(val) => val.residuals(),
            None => return Ok(parent_node),
        };

        let right_child =
            match self.compute_best_node(x_train, &available_columns, right_child_residuals) {
                Ok(val) => val,
                Err(error) => return Err(error.to_string()),
            };

        // Add to the parent
        parent_node.set_left_child(left_child);
        parent_node.set_right_child(right_child);

        Ok(parent_node)
    }

    fn compute_best_node(
        &self,
        x_train: &DataFrame,
        available_columns: &Vec<String>,
        residuals: &Vec<f64>,
    ) -> Result<Node, String> {
        // For each column, create a node
        let mut nodes = vec![];
        for i in 0..available_columns.len() {
            let values = match x_train.get_column(&available_columns[i][..]) {
                Some(values) => values,
                None => {
                    return Err(format!(
                        "Invalid feature column: {:?}",
                        available_columns[i]
                    ))
                }
            };

            let root_node =
                self.calculate_best_parameter_split(&values, &residuals, self.regularisation, i);
            nodes.push(root_node);
        }

        // Select best node based on the gain
        let best_node = self.select_best_gain(nodes);

        Ok(best_node)
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

    fn calculate_best_parameter_split(
        &self,
        values: &Vec<f64>,
        residuals: &Vec<f64>,
        regularisation: f64,
        column_index: usize,
    ) -> Node {
        // Sort values
        let (sorted_values, sorted_residuals) = self.sort_multiple(values, residuals);

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
        let mut nodes = vec![];
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
            let mut parent_node = Node::new(
                Some(column_index),
                Some(adjacent_average),
                sorted_residuals.clone(),
                regularisation,
            );

            parent_node.set_left_child(left_node);
            parent_node.set_right_child(right_node);

            nodes.push(parent_node);
        }

        self.select_best_gain(nodes)
    }

    fn select_best_gain(&self, mut nodes: Vec<Node>) -> Node {
        let mut best_index = 0;
        let mut best_gain = nodes[0].gain();
        for i in 0..nodes.len() {
            if i == 0 {
                continue;
            }

            let gain = nodes[i].gain();

            if gain > best_gain {
                best_index = i;
                best_gain = gain;
            }
        }

        let best_node = mem::replace(&mut nodes[best_index], Node::new(None, None, vec![], 0.0));

        best_node
    }

    /// Given two vectors, sorts vec1, and then reorganises vec2 by the new order of vec1
    fn sort_multiple(&self, vec1: &Vec<f64>, vec2: &Vec<f64>) -> (Vec<f64>, Vec<f64>) {
        // TODO: implementation
        (vec1.clone(), vec2.clone())
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
