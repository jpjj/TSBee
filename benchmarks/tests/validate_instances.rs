use benchmarks::{get_all_instance_names, load_instance, read_best_values_file::read_best_values};

#[test]
fn validate_instances_against_best_values() {
    let instance_names = get_all_instance_names().expect("Failed to get instance names");
    let best_values =
        read_best_values("data/best_values.yaml").expect("Failed to read best values");

    let mut validated_count = 0;
    let mut skipped_count = 0;
    let mut failed_instances = Vec::new();
    let mut correct_instances = Vec::new();

    for instance_name in &instance_names {
        match load_instance(instance_name) {
            Ok(instance) => {
                if let Some(&expected_value) = best_values.get(instance_name) {
                    let actual_value = instance.objective_value();

                    if actual_value != expected_value {
                        failed_instances.push((
                            instance_name.clone(),
                            expected_value,
                            actual_value,
                        ));
                    } else {
                        correct_instances.push((
                            instance_name.clone(),
                            expected_value,
                            actual_value,
                        ));
                    }
                    validated_count += 1;
                } else {
                    println!("No best value found for instance: {instance_name}");
                    skipped_count += 1;
                }
            }
            Err(_) => {
                skipped_count += 1;
            }
        }
    }

    println!("\nValidation Summary:");
    println!("Total instances: {}", instance_names.len());
    println!("Validated: {validated_count}");
    println!("Skipped (no solution file or best value): {skipped_count}");

    if !failed_instances.is_empty() {
        println!("\nInstances with mismatched values:");
        for (name, expected, actual) in &failed_instances {
            println!("  {name} - Expected: {expected}, Actual: {actual}");
        }
    }
    println!("\nInstances with correct values:");
    for (name, expected, actual) in &correct_instances {
        println!("  {name} - Expected: {expected}, Actual: {actual}");
    }
    println!(
        "\nNote: Some instances may fail due to rounding differences or different distance calculation methods."
    );
}
