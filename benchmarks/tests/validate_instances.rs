use benchmarks::{get_all_instance_names, load_instance, read_best_values_file::read_best_values};

#[test]
fn validate_instances_against_best_values() {
    let instance_names = get_all_instance_names().expect("Failed to get instance names");
    let best_values =
        read_best_values("data/best_values.yaml").expect("Failed to read best values");
    let mut failed_instances = Vec::new();
    let mut correct_instances = Vec::new();

    for instance_name in &instance_names {
        if let Ok(instance) = load_instance(instance_name) {
            if let Some(&expected_value) = best_values.get(instance_name) {
                let actual_value = instance.objective_value();

                if actual_value != expected_value {
                    failed_instances.push((instance_name.clone(), expected_value, actual_value));
                } else {
                    correct_instances.push((instance_name.clone(), expected_value, actual_value));
                }
            }
        }
    }

    assert!(failed_instances.is_empty())
}
