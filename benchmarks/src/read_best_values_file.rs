use serde_yaml;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

pub fn read_best_values<P: AsRef<Path>>(
    path: P,
) -> Result<HashMap<String, i64>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let yaml_map: serde_yaml::Value = serde_yaml::from_reader(reader)?;

    let mut best_values = HashMap::new();

    if let serde_yaml::Value::Mapping(map) = yaml_map {
        for (key, value) in map {
            if let (serde_yaml::Value::String(instance_name), serde_yaml::Value::Number(val)) =
                (key, value)
            {
                if let Some(num) = val.as_i64() {
                    best_values.insert(instance_name, num);
                }
            }
        }
    }

    Ok(best_values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_best_values() {
        let result = read_best_values("data/best_values.yaml");
        assert!(result.is_ok());

        let best_values = result.unwrap();

        assert_eq!(best_values.get("a280"), Some(&2579));
        assert_eq!(best_values.get("berlin52"), Some(&7542));
        assert_eq!(best_values.get("att48"), Some(&10628));

        assert!(best_values.contains_key("ali535"));
        assert!(best_values.contains_key("usa13509"));
    }
}
