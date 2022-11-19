mod objects;
use crate::objects::DataFrame;

fn main() {
    let x_train = [
        1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
    ]
    .to_vec();

    let columns = vec!["col1".to_string(), "col2".to_string(), "col3".to_string()];

    let dataframe_labelled = DataFrame::new_labelled(columns, x_train);

    println!("Labelled: {:?}", dataframe_labelled);

    let col2 = dataframe_labelled.get_column("col2");
    
    print!("col2: {:?}", col2);
}
