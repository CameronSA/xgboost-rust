use std::mem;

use crate::objects::Node;
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
        self,
        training_data: &DataFrame,
        target_column: &str,
        feature_columns: &Vec<String>,
    ) -> Result<XGBoost, String> {
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

        // Calculate root node
        let root_node = match self.compute_best_node(&x_train_with_residuals, &residuals_column) {
            Some(root_node) => root_node,
            None => return Err(format!("Failed to calculate root node")),
        };

        let decision_tree =
            self.build_decision_tree(x_train_with_residuals, &residuals_column, 0, root_node);

        println!("{:?}", decision_tree);

        Ok(self)
    }

    fn build_decision_tree(
        &self,
        dataset: DataFrame,
        residuals_column: &String,
        mut depth: i32,
        mut current_node: Node,
    ) -> Node {
        // Check depth
        if depth >= self.max_tree_depth {
            return current_node;
        }

        depth += 1;

        // Check sample size
        if dataset.len() < self.min_samples {
            return current_node;
        }

        // Build next level, implicitly checking gain at each.
        // If there is a child, that means we did a successful split on the previous step, so we can replace it with another parent + 2 children group
        current_node = match current_node.left_child() {
            Some(left_child) => {
                let child = self.build_decision_tree(
                    left_child.dataset().clone(),
                    residuals_column,
                    depth,
                    *left_child.clone(),
                );

                current_node.set_left_child(Some(child))
            }
            None => current_node,
        };

        current_node = match current_node.right_child() {
            Some(right_child) => {
                let child = self.build_decision_tree(
                    right_child.dataset().clone(),
                    residuals_column,
                    depth,
                    *right_child.clone(),
                );

                current_node.set_right_child(Some(child))
            }
            None => current_node,
        };

        current_node
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
