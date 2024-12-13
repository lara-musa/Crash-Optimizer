# Crash Optimizer

## ✩ Overview

This makes use of a Simulated Annealing algorithm to optimise project schedules, the code is meant to minimize project duration and cost while maximizing potential bonuses. The implementation allows for complex project scheduling with multiple execution methods per task, crash duration options, and dependency constraints. The Annealing algorithm is meant to make the process less computationally intensive but this comes at the cost of the output not being the absolute most optimal crashing method. Regardless, you still get a near-optimal high quality resut.

## ⊹ Why This Solution?

Traditional project scheduling approaches often struggle with:
* Time-cost trade off
* Handling multiple execution methods for tasks
* Considering complex task dependencies
* Optimizing for early completion bonuses

This code based approach resolves all these limitations in an approach that is not too computationally intensive.

## ➜ Usage

* Simply run this code in Python 3.9.6
* Since this program doesnt have a user interface yet, you'll have to edit certain parts of the code to suit your problem parameters.
  
## 𖦹 Customization

To adapt this optimizer to your specific project, modify these key parameters:

1. **Task Definition**:
   ```python
   tasks = [
       Task('TaskID', [
           {
               'normal_duration': int,     # Standard task duration
               'normal_cost': int,         # Cost at standard duration
               'max_crash_duration': int,  # Maximum days that can be reduced
               'crash_cost_per_day': int   # Cost to reduce duration by one day
           },
           # Additional method variations...
       ], [predecessor_task_ids])
   ]
   ```

2. **Optimization Parameters**:
   ```python
   SimulatedAnnealingProjectOptimizer(
       tasks=tasks,
       max_project_duration=int,      # Maximum allowed project duration
       budget=int,                    # Maximum crash budget
       bonus_schedule={               # Bonus for early completion
           days_early: bonus_amount,
           # Example: {10: 34925, 20: 84925}
       },
       initial_temperature=1000,      # Starting temperature for exploration
       cooling_rate=0.95,             # Rate of reducing exploration
       iterations=10000               # Number of optimization iterations
   )
   ```

## ⌕ Troubleshooting

* Ensure accurate task duration and cost estimates
* Define realistic crash duration and cost parameters
* Set a reasonable maximum project duration
* Calibrate bonus schedule to reflect project economics
* Adjust simulated annealing parameters based on project complexity

## Potential Future Enhancements

* Integrate with project management tools
* Add more sophisticated bonus calculation methods
* Implement additional constraint handling
* Create visualization of optimization process

## Dependencies

* NumPy
* Random
* Math

## License
Creative Commons CC-BY-NC-4.0
