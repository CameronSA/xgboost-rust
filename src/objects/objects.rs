#[derive(Debug)]
pub struct DataFrame {
    column_names: Vec<String>,
    rows: Vec<Vec<f64>>,
}

impl DataFrame {
    /// Create a labelled DataFrame with the given columns
    pub fn new_labelled(columns: Vec<String>, data_matrix: Vec<f64>) -> DataFrame {
        let n_columns = columns.len();
        let rows = Self::generate(n_columns, data_matrix);
        DataFrame {
            column_names: columns,
            rows: rows,
        }
    }

    /// Create an unlabelled DataFrame with the given number of columns
    pub fn new_unlabelled(n_columns: usize, data_matrix: Vec<f64>) -> DataFrame {
        let rows = Self::generate(n_columns, data_matrix);
        let mut columns = vec![];
        for i in 0..n_columns {
            columns.push(i.to_string())
        }

        DataFrame {
            column_names: columns,
            rows: rows,
        }
    }

    /// If the given column exists, returns the corresponding row items in that column.
    pub fn get_column(&self, column: &str) -> Option<Vec<f64>> {
        let column_index_option = &self.column_names.iter().position(|col| col == column);

        match column_index_option {
            Some(index) => Some(self.get_column_row_items(*index)),
            None => None
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
