#[derive(Clone, Debug)]
pub struct DataFrame {
    column_names: Vec<String>,
    rows: Vec<Vec<f64>>,
}

impl DataFrame {
    /// Create a labelled DataFrame with the given columns
    pub fn new_labelled(columns: Vec<String>, data_matrix: Vec<f64>) -> DataFrame {
        let n_columns = columns.len();
        let rows = DataFrame::generate(n_columns, data_matrix);
        DataFrame {
            column_names: columns,
            rows: rows,
        }
    }

    /// Create an unlabelled DataFrame with the given number of columns
    pub fn new_unlabelled(n_columns: usize, data_matrix: Vec<f64>) -> DataFrame {
        let rows = DataFrame::generate(n_columns, data_matrix);
        let mut columns = vec![];
        for i in 0..n_columns {
            columns.push(i.to_string())
        }

        DataFrame {
            column_names: columns,
            rows: rows,
        }
    }

    pub fn new_empty() -> DataFrame {
        DataFrame {
            column_names: vec![],
            rows: vec![],
        }
    }

    pub fn get_row(&self, row_index: usize) -> Result<&Vec<f64>, String> {
        // TODO: test
        if row_index >= self.rows.len() {
            return Err(String::from("Row index out of bounds"));
        }

        Ok(&self.rows[row_index])
    }

    pub fn add_row(mut self, row: Vec<f64>) -> DataFrame {
        // TODO: test
        self.rows.push(row);
        self
    }

    pub fn add_column(
        mut self,
        column_name: &str,
        column_values: Vec<f64>,
    ) -> Result<DataFrame, String> {
        // TODO: test

        if column_values.len() != self.rows.len() {
            return Err(String::from(format!(
                "The number of column values: {} does not match the number of rows: {}",
                column_values.len(),
                self.rows.len()
            )));
        }

        self.column_names.push(column_name.to_string());

        for i in 0..column_values.len() {
            self.rows[i].push(column_values[i]);
        }

        Ok(self)
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// If the given column exists, returns the corresponding row items in that column.
    pub fn get_column(&self, column: &str) -> Option<Vec<f64>> {
        let column_index_option = &self.column_names.iter().position(|col| col == column);

        match column_index_option {
            Some(index) => Some(self.get_column_row_items(*index)),
            None => None,
        }
    }

    /// If the given columns exist, returns a new dataframe with the given columns
    pub fn get_columns(&self, columns: &Vec<String>) -> Option<DataFrame> {
        let mut column_indices = vec![];
        for given_column in columns {
            for i in 0 as usize..self.column_names.len() {
                if given_column == &self.column_names[i] {
                    column_indices.push(i as i32);
                    break;
                }
            }
        }

        if column_indices.len() < 1 {
            return None;
        }

        let mut data_matrix = vec![];

        for row in &self.rows {
            for column_index in column_indices.iter() {
                data_matrix.push(row[*column_index as usize]);
            }
        }

        let mut valid_columns = vec![];
        for index in column_indices.iter() {
            valid_columns.push(self.column_names[*index as usize].to_string());
        }

        Some(DataFrame::new_labelled(valid_columns, data_matrix))
    }

    pub fn average(values: &Vec<f64>) -> f64 {
        let sum = values.iter().sum::<f64>();

        sum / values.len() as f64
    }

    pub fn column_names(&self) -> &Vec<String> {
        &self.column_names
    }

    fn get_column_row_items(&self, index: usize) -> Vec<f64> {
        let mut column = vec![];

        for row in &self.rows {
            column.push(row[index]);
        }

        return column;
    }

    /// From the given data matrix, turns it into a vector of rows where each row has n_columns columns
    fn generate(n_columns: usize, data_matrix: Vec<f64>) -> Vec<Vec<f64>> {
        let mut rows = vec![];
        let mut i = 0;
        while i < data_matrix.len() {
            let mut row = vec![];
            for n in i..n_columns + i {
                row.push(data_matrix[n]);
            }

            rows.push(row);
            i += n_columns;
        }

        rows
    }
}

/// A node with no children, column index or split value is a leaf. Column indices and split values refer to child nodes
#[derive(Clone, Debug)]
pub struct Node {
    column_index: Option<usize>,
    column_split_value: Option<f64>,
    dataset: DataFrame,
    residuals_column: String,
    regularisation: f64,
    left_child: Option<Box<Node>>,
    right_child: Option<Box<Node>>,
}

impl Node {
    pub fn new(
        column_index: Option<usize>,
        column_split_value: Option<f64>,
        dataset: DataFrame,
        residuals_column: String,
        regularisation: f64,
    ) -> Self {
        Node {
            column_index,
            column_split_value,
            dataset,
            residuals_column,
            regularisation,
            left_child: None,
            right_child: None,
        }
    }

    pub fn column_index(&self) -> Option<usize> {
        self.column_index
    }

    pub fn column_split_value(&self) -> Option<f64> {
        self.column_split_value
    }

    pub fn dataset(&self) -> &DataFrame {
        &self.dataset
    }

    pub fn residuals_column(&self) -> &String {
        &self.residuals_column
    }

    pub fn regularisation(&self) -> f64 {
        self.regularisation
    }

    pub fn left_child(&self) -> &Option<Box<Node>> {
        &self.left_child
    }

    pub fn right_child(&self) -> &Option<Box<Node>> {
        &self.right_child
    }

    pub fn set_left_child(mut self, left_child: Option<Node>) -> Self {
        self.left_child = match left_child {
            Some(node) => Some(Box::new(node)),
            Node => None,
        };

        self
    }

    pub fn set_right_child(mut self, right_child: Option<Node>) -> Self {
        self.right_child = match right_child {
            Some(node) => Some(Box::new(node)),
            Node => None,
        };

        self
    }

    /// The similarity score is the sum of the residuals squared, divided by the number of residuals plus the regularisation parameter
    pub fn similarity_score(&self) -> f64 {
        let residuals = match self.dataset.get_column(&self.residuals_column) {
            Some(val) => val,
            None => return 0.0,
        };

        let sum = residuals.iter().sum::<f64>();

        let score = (sum * sum) / (residuals.len() as f64 + self.regularisation);

        score
    }

    // Calculates the gain of the child split. If there are no children, returns 0
    pub fn gain(&self) -> f64 {
        let parent_similarity_score = &self.similarity_score();

        let left_child_similarity_score = match &self.left_child {
            Some(child) => child.similarity_score(),
            None => return 0.0,
        };

        let right_child_similarity_score = match &self.right_child {
            Some(child) => child.similarity_score(),
            None => return 0.0,
        };

        let gain =
            left_child_similarity_score + right_child_similarity_score - parent_similarity_score;

        gain
    }
}

#[cfg(test)]
mod tests {
    use crate::objects::DataFrame;

    use super::Node;

    fn test_df() -> DataFrame {
        let x_train = [1.0, 2.0, 3.0].to_vec();

        let columns = vec!["col1".to_string(), "col2".to_string(), "col3".to_string()];

        let dataframe_labelled = DataFrame::new_labelled(columns, x_train);

        dataframe_labelled
    }

    #[test]
    fn test_add_nodes() {
        let mut node = Node::new(Some(1), Some(1.0), test_df(), "col1".to_string(), 1.0);
        node = node.set_left_child(Some(Node::new(
            Some(2),
            Some(1.0),
            test_df(),
            "col1".to_string(),
            1.0,
        )));
        node = node.set_right_child(Some(Node::new(
            Some(3),
            Some(1.0),
            test_df(),
            "col1".to_string(),
            1.0,
        )));

        let expected = r#"Node { column_index: Some(1), column_split_value: Some(1.0), dataset: DataFrame { column_names: ["col1", "col2", "col3"], rows: [[1.0, 2.0, 3.0]] }, residuals_column: "col1", regularisation: 1.0, left_child: Some(Node { column_index: Some(2), column_split_value: Some(1.0), dataset: DataFrame { column_names: ["col1", "col2", "col3"], rows: [[1.0, 2.0, 3.0]] }, residuals_column: "col1", regularisation: 1.0, left_child: None, right_child: None }), right_child: Some(Node { column_index: Some(3), column_split_value: Some(1.0), dataset: DataFrame { column_names: ["col1", "col2", "col3"], rows: [[1.0, 2.0, 3.0]] }, residuals_column: "col1", regularisation: 1.0, left_child: None, right_child: None }) }"#;

        let actual = format!("{:?}", node);

        assert_eq!(expected, actual);
    }

    #[test]
    fn test_get_nodes() {
        let mut node = Node::new(Some(1), Some(1.0), test_df(), "col1".to_string(), 1.0);
        node = node.set_left_child(Some(Node::new(
            Some(2),
            Some(1.0),
            test_df(),
            "col1".to_string(),
            1.0,
        )));
        node = node.set_right_child(Some(Node::new(
            Some(3),
            Some(1.0),
            test_df(),
            "col1".to_string(),
            1.0,
        )));

        let left = match node.left_child() {
            Some(left) => left,
            None => panic!("left is None"),
        };

        let right = match node.right_child() {
            Some(right) => right,
            None => panic!("left is None"),
        };

        let expected_left = r#"Node { column_index: Some(2), column_split_value: Some(1.0), dataset: DataFrame { column_names: ["col1", "col2", "col3"], rows: [[1.0, 2.0, 3.0]] }, residuals_column: "col1", regularisation: 1.0, left_child: None, right_child: None }"#;
        let expected_right = r#"Node { column_index: Some(3), column_split_value: Some(1.0), dataset: DataFrame { column_names: ["col1", "col2", "col3"], rows: [[1.0, 2.0, 3.0]] }, residuals_column: "col1", regularisation: 1.0, left_child: None, right_child: None }"#;

        let actual_left = format!("{:?}", left);

        let actual_right = format!("{:?}", right);

        assert_eq!(expected_left.to_string(), actual_left);
        assert_eq!(expected_right.to_string(), actual_right);
    }

    #[test]
    fn test_get_column_index() {
        let node = Node::new(Some(1), Some(1.1), test_df(), "col1".to_string(), 1.2);

        let column_index = node.column_index();

        assert_eq!(column_index, Some(1));
    }

    #[test]
    fn test_get_column_split_value() {
        let node = Node::new(Some(1), Some(1.1), test_df(), "col1".to_string(), 1.2);

        let column_split_value = node.column_split_value();

        assert_eq!(column_split_value, Some(1.1));
    }

    #[test]
    fn test_get_residuals_column() {
        let node = Node::new(Some(1), Some(1.1), test_df(), "col1".to_string(), 1.2);

        let residuals = node.residuals_column().to_string();

        assert_eq!(residuals, "col1".to_string());
    }

    #[test]
    fn test_get_regularisation() {
        let node = Node::new(Some(1), Some(1.1), test_df(), "col1".to_string(), 1.2);

        let regularisation = node.regularisation();

        assert_eq!(regularisation, 1.2);
    }

    #[test]
    fn test_get_dataset() {
        let node = Node::new(Some(1), Some(1.1), test_df(), "col1".to_string(), 1.2);

        let dataset = node.dataset();

        let expected_str =
            r#"DataFrame { column_names: ["col1", "col2", "col3"], rows: [[1.0, 2.0, 3.0]] }"#;

        assert_eq!(format!("{:?}", dataset), expected_str);
    }
}

#[cfg(test)]
mod dataframe_tests {
    use super::DataFrame;

    #[test]
    fn test_new_labelled() {
        let data_matrix = [
            1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
        ]
        .to_vec();

        let columns = ["col1", "col2", "col3"].map(String::from).to_vec();

        let expected_columns = columns.clone();

        let df = DataFrame::new_labelled(columns, data_matrix);

        assert_eq!(expected_columns, df.column_names);
        assert_eq!([1.0, 1.0, 1.0].to_vec(), df.rows[0]);
        assert_eq!([1.0, 1.0, 0.0].to_vec(), df.rows[1]);
        assert_eq!([1.0, 1.0, 1.0].to_vec(), df.rows[2]);
        assert_eq!([0.0, 0.0, 0.0].to_vec(), df.rows[3]);
        assert_eq!([1.0, 1.0, 1.0].to_vec(), df.rows[4]);
    }

    #[test]
    fn test_new_unlabelled() {
        let data_matrix = [
            1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
        ]
        .to_vec();

        let df = DataFrame::new_unlabelled(3, data_matrix);

        assert_eq!(["0", "1", "2"].map(String::from).to_vec(), df.column_names);
        assert_eq!([1.0, 1.0, 1.0].to_vec(), df.rows[0]);
        assert_eq!([1.0, 1.0, 0.0].to_vec(), df.rows[1]);
        assert_eq!([1.0, 1.0, 1.0].to_vec(), df.rows[2]);
        assert_eq!([0.0, 0.0, 0.0].to_vec(), df.rows[3]);
        assert_eq!([1.0, 1.0, 1.0].to_vec(), df.rows[4]);
    }

    #[test]
    fn test_get_column_column_exists() {
        let data_matrix = [
            1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
        ]
        .to_vec();

        let columns = ["col1", "col2", "col3"].map(String::from).to_vec();

        let df = DataFrame::new_labelled(columns, data_matrix);

        let column_vals = df.get_column("col2").unwrap();

        assert_eq!([1.0, 1.0, 1.0, 0.0, 1.0].to_vec(), column_vals);
    }

    #[test]
    fn test_get_column_column_does_not_exist() {
        let data_matrix = [
            1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
        ]
        .to_vec();

        let columns = ["col1", "col2", "col3"].map(String::from).to_vec();

        let df = DataFrame::new_labelled(columns, data_matrix);

        let column_vals = df.get_column("invalid");

        assert_eq!(None, column_vals);
    }

    #[test]
    fn test_get_columns_columns_exist() {
        let data_matrix = [
            1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
        ]
        .to_vec();

        let columns = ["col1", "col2", "col3"].map(String::from).to_vec();

        let df = DataFrame::new_labelled(columns, data_matrix);

        let column_vals = match df.get_columns(&["col2", "col3"].map(String::from).to_vec()) {
            Some(column_vals) => column_vals,
            None => panic!("Could not find columns"),
        };

        let col2_vals = column_vals.get_column("col2").unwrap();
        let col3_vals = column_vals.get_column("col3").unwrap();

        assert_eq!([1.0, 1.0, 1.0, 0.0, 1.0].to_vec(), col2_vals);
        assert_eq!([1.0, 0.0, 1.0, 0.0, 1.0].to_vec(), col3_vals);
    }

    #[test]
    fn test_get_columns_column_does_not_exist() {
        let data_matrix = [
            1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
        ]
        .to_vec();

        let columns = ["col1", "col2", "col3"].map(String::from).to_vec();

        let df = DataFrame::new_labelled(columns, data_matrix);

        let column_vals = df.get_columns(&["invalid", "col3"].map(String::from).to_vec());

        let column_vals = match df.get_columns(&["invalid", "col3"].map(String::from).to_vec()) {
            Some(column_vals) => column_vals,
            None => panic!("Could not find columns"),
        };

        let col3_vals = column_vals.get_column("col3").unwrap();

        assert_eq!(
            column_vals.column_names(),
            &["col3"].map(String::from).to_vec()
        );
        assert_eq!([1.0, 0.0, 1.0, 0.0, 1.0].to_vec(), col3_vals);
    }

    #[test]
    fn test_average() {
        let values = vec![1.2, 3.5, 6.2, 3.0];
        let average = DataFrame::average(&values);

        assert_eq!(3.475, average);
    }

    #[test]
    fn test_column_names() {
        let data_matrix = [
            1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
        ]
        .to_vec();

        let columns = ["col1", "col2", "col3"].map(String::from).to_vec();

        let df = DataFrame::new_labelled(columns, data_matrix);
        assert_eq!(
            df.column_names(),
            &["col1", "col2", "col3"].map(String::from).to_vec()
        );
    }
}
