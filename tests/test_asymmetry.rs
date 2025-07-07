use chrono::TimeDelta;
use tsbee::domain::city::City;
use tsbee::domain::route::Route;
use tsbee::input::Input;
use tsbee::penalties::distance::{DistanceMatrix, DistancePenalizer};
use tsbee::solver::Solver;
#[test]
fn test_br17() {
    let matrix = vec![
        vec![0, 3, 5, 48, 48, 8, 8, 5, 5, 3, 3, 0, 3, 5, 8, 8, 5],
        vec![3, 0, 3, 48, 48, 8, 8, 5, 5, 0, 0, 3, 0, 3, 8, 8, 5],
        vec![5, 3, 0, 72, 72, 48, 48, 24, 24, 3, 3, 5, 3, 0, 48, 48, 24],
        vec![48, 48, 74, 0, 0, 6, 6, 12, 12, 48, 48, 48, 48, 74, 6, 6, 12],
        vec![48, 48, 74, 0, 0, 6, 6, 12, 12, 48, 48, 48, 48, 74, 6, 6, 12],
        vec![8, 8, 50, 6, 6, 0, 0, 8, 8, 8, 8, 8, 8, 50, 0, 0, 8],
        vec![8, 8, 50, 6, 6, 0, 0, 8, 8, 8, 8, 8, 8, 50, 0, 0, 8],
        vec![5, 5, 26, 12, 12, 8, 8, 0, 0, 5, 5, 5, 5, 26, 8, 8, 0],
        vec![5, 5, 26, 12, 12, 8, 8, 0, 0, 5, 5, 5, 5, 26, 8, 8, 0],
        vec![3, 0, 3, 48, 48, 8, 8, 5, 5, 0, 0, 3, 0, 3, 8, 8, 5],
        vec![3, 0, 3, 48, 48, 8, 8, 5, 5, 0, 0, 3, 0, 3, 8, 8, 5],
        vec![0, 3, 5, 48, 48, 8, 8, 5, 5, 3, 3, 0, 3, 5, 8, 8, 5],
        vec![3, 0, 3, 48, 48, 8, 8, 5, 5, 0, 0, 3, 0, 3, 8, 8, 5],
        vec![5, 3, 0, 72, 72, 48, 48, 24, 24, 3, 3, 5, 3, 0, 48, 48, 24],
        vec![8, 8, 50, 6, 6, 0, 0, 8, 8, 8, 8, 8, 8, 50, 0, 0, 8],
        vec![8, 8, 50, 6, 6, 0, 0, 8, 8, 8, 8, 8, 8, 50, 0, 0, 8],
        vec![5, 5, 26, 12, 12, 8, 8, 0, 0, 5, 5, 5, 5, 26, 8, 8, 0],
    ];
    let mut distance_matrix = DistanceMatrix::new(matrix);
    let penalizer = DistancePenalizer::new(distance_matrix.clone());
    let optimal_tour = Route::new(
        [6, 15, 16, 5, 4, 8, 17, 9, 1, 12, 3, 14, 10, 11, 2, 13, 7]
            .iter()
            .map(|x| City(x - 1))
            .collect(),
    );
    assert_eq!(penalizer.penalize(&optimal_tour).distance, 39);
    let big_m = distance_matrix.sum_of_abs_distance();
    distance_matrix = distance_matrix.symmetrize();
    let input = Input::new(distance_matrix, Some(TimeDelta::seconds(1)));
    let mut solver = Solver::new(input);
    let mut solution_report = solver.solve(false);
    solution_report = solution_report.desymmetrize(big_m);
    assert_eq!(
        penalizer
            .penalize(&solution_report.best_solution.route)
            .distance,
        39
    );
}
