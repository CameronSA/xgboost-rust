use std::mem;

use super::node::Node;
use crate::objects::DataFrame;
use uuid::Uuid;

pub struct XGBoost {
    learning_rate: f64,
    n_estimators: i32,
    regularisation: f64,
    max_tree_depth: i32,
    min_gain: f64,
    min_samples: usize,
}

impl XGBoost {
    pub fn new(
        learning_rate: f64,
        n_estimators: i32,
        regularisation: f64,
        max_tree_depth: i32,
        min_gain: f64,
        min_samples: usize,
    ) -> Self {
        XGBoost {
            learning_rate,
            n_estimators,
            regularisation,
            max_tree_depth,
            min_gain,
            min_samples,
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
        let residuals = match Self::residuals(&y_train, &vec![initial_prediction; y_train.len()]) {
            Ok(val) => val,
            Err(error) => return Err(error.to_string()),
        };

        // Add residuals to dataset
        let residuals_column = format!("residuals_{}", Uuid::new_v4());
        let x_train_with_residuals = match x_train.add_column(&residuals_column, residuals) {
            Ok(val) => val,
            Err(error) => return Err(error.to_string()),
        };

        let decision_tree =
            self.build_decision_tree(x_train_with_residuals, &residuals_column, 0, None);

        Ok(true)
    }

    fn build_decision_tree(
        &self,
        dataset: DataFrame,
        residuals_column: &String,
        mut depth: i32,
        mut current_node: Option<Node>,
    ) -> Option<Node> {
        // Check depth
        if depth >= self.max_tree_depth {
            return current_node;
        }

        depth += 1;

        // Check sample size
        if dataset.len() < self.min_samples {
            return current_node;
        }

        // Get the current node. If first iteration (i.e. root node is none), calculate a root node.
        let mut current_node = match current_node {
            Some(node) => node,
            None => match self.compute_best_node(&dataset, &residuals_column) {
                Some(root_node) => root_node,
                None => return None,
            },
        };

        // Build next level, implicitly checking gain at each.
        // If there is a child, that means we did a successful split on the previous step, so we can replace it with another parent + 2 children group
        current_node = match current_node.left_child() {
            Some(left_node) => {
                let node = current_node
                    .clone()
                    .set_left_child(self.build_decision_tree(
                        left_node.dataset().clone(),
                        residuals_column,
                        depth,
                        Some(current_node),
                    ));

                node
            }
            None => current_node,
        };

        current_node = match current_node.right_child() {
            Some(right_node) => {
                let node = current_node
                    .clone()
                    .set_right_child(self.build_decision_tree(
                        right_node.dataset().clone(),
                        residuals_column,
                        depth,
                        Some(current_node),
                    ));
                node
            }
            None => current_node,
        };

        Some(current_node)
    }

    fn compute_best_node(&self, dataset: &DataFrame, residuals_column: &String) -> Option<Node> {
        let available_columns = dataset.column_names();

        // For each column, create a node
        let mut nodes = vec![];
        for i in 0..available_columns.len() {
            if available_columns[i] == residuals_column.to_string() {
                continue;
            }

            // These panics should never happen
            let values = match dataset.get_column(&available_columns[i][..]) {
                Some(values) => values,
                None => panic!("Couldn't find column"),
            };

            let residuals = match dataset.get_column(&residuals_column[..]) {
                Some(val) => val,
                None => panic!("Couldn't find residuals"),
            };

            match self.calculate_best_parameter_split(
                dataset.clone(),
                residuals_column,
                &available_columns[i],
                self.regularisation,
                i,
            ) {
                Some(val) => nodes.push(val),
                None => (),
            };
        }

        if nodes.len() == 0 {
            return None;
        }

        // Select best gain
        self.select_best_gain(nodes)
    }

    /// If None: gain condition was not met
    fn calculate_best_parameter_split(
        &self,
        dataset: DataFrame,
        residuals_column: &String,
        split_column: &String,
        regularisation: f64,
        column_index: usize,
    ) -> Option<Node> {
        // Sort values
        let sorted_dataset = self.sort_dataframe(dataset, split_column);

        let sorted_values = match sorted_dataset.get_column(&split_column[..]) {
            Some(values) => values,
            None => panic!("Couldn't find column"),
        };

        // Find average of each of the adjacent values
        let mut adjacent_averages = vec![];
        for i in 0..sorted_dataset.len() {
            if i == sorted_dataset.len() - 1 {
                break;
            } else {
                let adjacent_average = (sorted_values[i] + sorted_values[i + 1]) / 2.0;
                adjacent_averages.push(adjacent_average);
            }
        }

        // For each adjacent average, create a node split
        let mut nodes = vec![];
        for adjacent_average in adjacent_averages {
            let mut left_dataset =
                DataFrame::new_labelled(sorted_dataset.column_names().clone(), vec![]);
            let mut right_dataset =
                DataFrame::new_labelled(sorted_dataset.column_names().clone(), vec![]);

            for i in 0..sorted_dataset.len() {
                let row = match sorted_dataset.get_row(i) {
                    Ok(row) => row,
                    Err(err) => panic!("{}", err),
                };

                if sorted_values[i] <= adjacent_average {
                    left_dataset = left_dataset.add_row(row.clone());
                } else {
                    right_dataset = right_dataset.add_row(row.clone());
                }
            }

            let left_node = Node::new(
                None,
                None,
                left_dataset,
                residuals_column.to_string(),
                regularisation,
            );
            let right_node = Node::new(
                None,
                None,
                right_dataset,
                residuals_column.to_string(),
                regularisation,
            );
            let mut parent_node = Node::new(
                Some(column_index),
                Some(adjacent_average),
                sorted_dataset.clone(),
                residuals_column.to_string(),
                regularisation,
            );

            parent_node = parent_node.set_left_child(Some(left_node));
            parent_node = parent_node.set_right_child(Some(right_node));

            nodes.push(parent_node);
        }

        // Select best gain
        self.select_best_gain(nodes)
    }

    fn residuals(
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

    fn select_best_gain(&self, mut nodes: Vec<Node>) -> Option<Node> {
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

        if best_gain < self.min_gain {
            return None;
        }

        let best_node = mem::replace(
            &mut nodes[best_index],
            Node::new(None, None, DataFrame::new_empty(), "".to_string(), 0.0),
        );

        Some(best_node)
    }

    /// Given a dataframe, sorts it by the given column
    fn sort_dataframe(&self, dataset: DataFrame, sort_column: &String) -> DataFrame {
        // TODO: implementation
        dataset
    }
}

// fn build_decision_tree(
//     &self,
//     x_train: &DataFrame,
//     feature_columns: &Vec<String>,
//     residuals: &Vec<f64>,
// ) -> Result<Node, String> {
//     let mut available_columns = feature_columns.clone();

//     // Calculate root node
//     let mut root_node = match self.compute_best_node(x_train, &available_columns, residuals) {
//         Ok(val) => val,
//         Err(error) => return Err(error.to_string()),
//     };

//     let root_column_index = match root_node.column_index() {
//         Some(val) => val,
//         None => return Ok(root_node),
//     };

//     // Whatever column was chosen for the root node, remove it from the column pool
//     available_columns.remove(root_column_index);

//     // NOTE: Just for the left (to get my head around it)
//     // Will need multiple instances of 'available_columns' for each path through the tree

//     // Iteration 1
//     let root_node = match self.add_children(root_node, x_train, &available_columns) {
//         Ok(val) => val,
//         Err(error) => return Err(error.to_string()),
//     };

//     let left_node = match root_node.left_child(){
//         Some(val) => val,
//         None => return Ok(root_node),
//     };

//     let left_node_column_index = match left_node.column_index() {
//         Some(val) => val,
//         None => return Ok(root_node),
//     };

//     available_columns.remove(left_node_column_index);

//     // Iteration 2
//     let root_node = match self.add_children(*left_node.clone(), x_train, &available_columns) {
//         Ok(val) => val,
//         Err(error) => return Err(error.to_string()),
//     };

//     let left_left_node = match left_node.left_child(){
//         Some(val) =>val,
//         None => return Ok(root_node),
//     };

//     let left_left_node_column_index = match left_left_node.column_index() {
//         Some(val) => val,
//         None => return Ok(root_node),
//     };

//     available_columns.remove(left_left_node_column_index);

//     // Iteration 3
//     // ...

//     Ok(root_node)
// }

// fn add_children(
//     &self,
//     mut parent_node: Node,
//     x_train: &DataFrame,
//     available_columns: &Vec<String>,
// ) -> Result<Node, String> {
//     // First the left
//     let left_child_residuals = match parent_node.left_child() {
//         Some(val) => val.residuals(),
//         None => return Ok(parent_node),
//     };

//     let left_child =
//         match self.compute_best_node(x_train, &available_columns, left_child_residuals) {
//             Ok(val) => val,
//             Err(error) => return Err(error.to_string()),
//         };

//     // Now the right
//     let right_child_residuals = match parent_node.right_child() {
//         Some(val) => val.residuals(),
//         None => return Ok(parent_node),
//     };

//     let right_child =
//         match self.compute_best_node(x_train, &available_columns, right_child_residuals) {
//             Ok(val) => val,
//             Err(error) => return Err(error.to_string()),
//         };

//     // Add to the parent
//     parent_node.set_left_child(left_child);
//     parent_node.set_right_child(right_child);

//     Ok(parent_node)
// }

// fn select_best_gain(&self, mut nodes: Vec<Node>) -> Node {
//     let mut best_index = 0;
//     let mut best_gain = nodes[0].gain();
//     for i in 0..nodes.len() {
//         if i == 0 {
//             continue;
//         }

//         let gain = nodes[i].gain();

//         if gain > best_gain {
//             best_index = i;
//             best_gain = gain;
//         }
//     }

//     let best_node = mem::replace(&mut nodes[best_index], Node::new(None, None, vec![], 0.0));

//     best_node
// }

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
