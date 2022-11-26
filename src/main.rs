mod objects;
mod algorithm;
use crate::objects::DataFrame;
use crate::algorithm::XGBoost;

fn main() {
    let x_train = [
        1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
    ]
    .to_vec();

    let columns = vec!["col1".to_string(), "col2".to_string(), "col3".to_string()];

    let dataframe_labelled = DataFrame::new_labelled(columns, x_train);

    let model = XGBoost::new(1.0,10,1.0,5,0.0,1);
    let train_result = model.train(&dataframe_labelled, "col3", &["col1".to_string(),"col2".to_string()].to_vec());
}
