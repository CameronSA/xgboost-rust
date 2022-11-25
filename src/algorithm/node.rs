/// A node with no children, column index or split value is a leaf. Column indices and split values refer to child nodes
#[derive(Clone, Debug)]
pub struct Node {
    column_index: Option<usize>,
    column_split_value: Option<f64>,
    residuals: Vec<f64>,
    regularisation: f64,
    left_child: Option<Box<Node>>,
    right_child: Option<Box<Node>>,
}

impl Node {
    pub fn new(
        column_index: Option<usize>,
        column_split_value: Option<f64>,
        residuals: Vec<f64>,
        regularisation: f64,
    ) -> Self {
        Node {
            column_index,
            column_split_value,
            residuals,
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

    pub fn residuals(&self) -> &Vec<f64> {
        &self.residuals
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

    pub fn set_left_child(&mut self, left_child: Node) {
        self.left_child = Some(Box::new(left_child));
    }

    pub fn set_right_child(&mut self, right_child: Node) {
        self.right_child = Some(Box::new(right_child));
    }

    /// The similarity score is the sum of the residuals squared, divided by the number of residuals plus the regularisation parameter
    pub fn similarity_score(&self) -> f64 {
        let sum = self.residuals.iter().sum::<f64>();

        let score = (sum * sum) / (self.residuals.len() as f64 + self.regularisation);

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
    use super::Node;

    #[test]
    fn test_add_nodes() {
        let mut node = Node::new(Some(1), Some(1.0), vec![1.0], 1.0);
        node.set_left_child(Node::new(Some(2), Some(1.0), vec![1.0], 1.0));
        node.set_right_child(Node::new(Some(3), Some(1.0), vec![1.0], 1.0));

        let expected = "Node { column_index: Some(1), column_split_value: Some(1.0), residuals: [1.0], regularisation: 1.0, left_child: Some(Node { column_index: Some(2), column_split_value: Some(1.0), residuals: [1.0], regularisation: 1.0, left_child: None, right_child: None }), right_child: Some(Node { column_index: Some(3), column_split_value: Some(1.0), residuals: [1.0], regularisation: 1.0, left_child: None, right_child: None }) }";

        let actual = format!("{:?}", node);

        assert_eq!(expected, actual);
    }

    #[test]
    fn test_get_nodes() {
        let mut node = Node::new(Some(1), Some(1.0), vec![1.0], 1.0);
        node.set_left_child(Node::new(Some(2), Some(1.0), vec![1.0], 1.0));
        node.set_right_child(Node::new(Some(3), Some(1.0), vec![1.0], 1.0));

        let left = match node.left_child() {
            Some(left) => left,
            None => panic!("left is None"),
        };

        let right = match node.right_child() {
            Some(right) => right,
            None => panic!("left is None"),
        };

        let expected_left = "Node { column_index: Some(2), column_split_value: Some(1.0), residuals: [1.0], regularisation: 1.0, left_child: None, right_child: None }";
        let expected_right = "Node { column_index: Some(3), column_split_value: Some(1.0), residuals: [1.0], regularisation: 1.0, left_child: None, right_child: None }";

        let actual_left = format!("{:?}", left);

        let actual_right = format!("{:?}", right);

        assert_eq!(expected_left.to_string(), actual_left);
        assert_eq!(expected_right.to_string(), actual_right);
    }

    #[test]
    fn test_get_column_index() {
        let node = Node::new(Some(1), Some(1.1), vec![1.0], 1.2);

        let column_index = node.column_index();

        assert_eq!(column_index, Some(1));
    }

    #[test]
    fn test_get_column_split_value() {
        let node = Node::new(Some(1), Some(1.1), vec![1.0], 1.2);

        let column_split_value = node.column_split_value();

        assert_eq!(column_split_value, Some(1.1));
    }

    #[test]
    fn test_get_residuals() {
        let node = Node::new(Some(1), Some(1.1), vec![1.0], 1.2);

        let residuals = node.residuals();

        assert_eq!(residuals, &vec![1.0]);
    }

    #[test]
    fn test_get_regularisation() {
        let node = Node::new(Some(1), Some(1.1), vec![1.0], 1.2);

        let regularisation = node.regularisation();

        assert_eq!(regularisation, 1.2);
    }
}
