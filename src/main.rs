mod objects;
use crate::objects::DataFrame;

fn main() {
    let x_train = [
        1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
    ]
    .to_vec();

    let columns = vec!["col1".to_string(), "col2".to_string(), "col3".to_string()];

    // let dataframe_labelled = DataFrame::new_labelled(columns, x_train);

    let dataframe_unlabelled = DataFrame::new_unlabelled(columns.len(), x_train);

    //println!("Labelled: {:?}", dataframe_labelled);
    println!("Unlabelled: {:?}", dataframe_unlabelled);
}
